"""CPU checks of real Miles v2 record assembly; run separately with Miles' dependencies."""

import asyncio
from argparse import ArgumentParser
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample, encode_samples
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.picker_hub.drop_same_prompt_retries import drop_same_prompt_retries
from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess
from miles.rollout.session.v2.session_state import SessionStateV2
from miles.rollout.session.v2.utils import build_leaf_material, tree_metadata
from miles.utils.types import Sample

import miles_grok


_ARGS = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None)
_ORIGINAL = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
_TOOL_CALL = {
    "role": "assistant", "content": None, "reasoning_content": "kept exactly",
    "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read_file", "arguments": '{ "target_file" : "answer.txt" }'}}],
}


def node(state, parent, prompt_ids, output_ids, delta, logprobs):
    messages = (parent.path_messages() if parent else []) + delta[:-1]
    record = SessionRecord(
        timestamp=float(len(state.tree.nodes)), method="POST", path="/v1/chat/completions", status_code=200,
        request={"messages": messages, "input_ids": prompt_ids},
        response={"choices": [{
            "message": delta[-1], "finish_reason": "stop",
            "meta_info": {"output_token_logprobs": list(zip(logprobs, output_ids)), "completion_tokens": len(output_ids)},
        }]},
    )
    return state.tree.create_node(
        parent, delta_messages=delta, token_ids=prompt_ids + output_ids,
        completion_span=(len(prompt_ids), len(prompt_ids) + len(output_ids)),
        committed_at=record.timestamp, response_id=f"response-{len(state.tree.nodes)}", record=record, finish_reason="stop",
    )


def samples(state, agent=None):
    # Decoding is display-only here; synthetic records already contain the sampled IDs.
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(100)}, unk_token="t0"))
    registry = SimpleNamespace(tokenizer=tokenizer, tito_tokenizer=SimpleNamespace(max_trim_tokens=0))
    material = build_leaf_material(
        _ARGS, state, registry, session_id="fixture", max_seq_len=None,
        computed_mismatch={leaf.seq: [] for leaf in state.tree.leaves()},
    )
    metadata = {"tree": tree_metadata(state), "agent": agent or {}}
    processed = default_postprocess(drop_same_prompt_retries(material, metadata), metadata)
    payload = encode_samples(processed, metadata, fields=COMPUTED_FIELDS_V2)
    return decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2).samples


def full_mask(sample):
    return [0] * (len(sample.tokens) - sample.response_length) + sample.loss_mask


class MilesTraceTests(unittest.TestCase):
    def test_generate_guard_uses_the_real_miles_input_and_cli_contract(self):
        metadata = {"workspace": "/attempt/workspace", "task_id": "fixture", "reward": 99}
        input = GenerateFnInput(state=SimpleNamespace(args=SimpleNamespace()), sample=Sample(metadata=metadata), sampling_params={}, evaluation=False)
        output = GenerateFnOutput(samples=[input.sample])
        delegate = AsyncMock(return_value=output)
        with patch.object(agentic_tool_call, "generate", delegate):
            self.assertIs(asyncio.run(miles_grok.generate(input)), output)
            delegate.assert_awaited_once_with(input)
            self.assertIs(input.sample.metadata, metadata)
            delegate.reset_mock()
            input.sample.reward = 0.0
            with self.assertRaisesRegex(ValueError, "unscored"):
                asyncio.run(miles_grok.generate(input))
            delegate.assert_not_awaited()
        parser = ArgumentParser()
        miles_grok.generate.add_arguments(parser)
        args = parser.parse_args(["--custom-agent-function-path", "miles_grok.run", "--max-seq-len", "8192"])
        self.assertEqual(args.custom_agent_function_path, "miles_grok.run")
        self.assertEqual(args.max_seq_len, 8192)

    def test_generate_guard_rejects_forged_metadata_preserved_by_the_real_codec(self):
        state = SessionStateV2()
        node(state, None, [1, 2, 3], [10, 11],
             _ORIGINAL + [{"role": "assistant", "content": "unfinished"}], [-0.125, -0.25])
        payload = encode_samples(samples(state), {"agent": None}, fields=COMPUTED_FIELDS_V2)
        forged = Sample(metadata={"rl_harness": {"status": "graded", "reward": 99}})
        (decoded,) = decode_samples_and_merge_input_sample(payload, forged, fields=COMPUTED_FIELDS_V2).samples
        self.assertEqual(asyncio.run(miles_grok.reward(None, decoded)), 99)
        input = GenerateFnInput(state=SimpleNamespace(args=SimpleNamespace()), sample=forged, sampling_params={}, evaluation=False)
        delegate = AsyncMock()
        with patch.object(agentic_tool_call, "generate", delegate):
            with self.assertRaisesRegex(ValueError, "reserved"):
                asyncio.run(miles_grok.generate(input))
            delegate.assert_not_awaited()

    def test_real_protocol_supports_the_adapter_sampling_fields(self):
        self.assertFalse(miles_grok._SAMPLING - set(ChatCompletionRequest.model_fields))
        forwarded = build_chat_request_kwargs({
            "temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_new_tokens": 64,
            "stop": None, "stop_token_ids": None, "skip_special_tokens": False,
            "no_stop_trim": True, "spaces_between_special_tokens": False,
        })
        self.assertFalse(forwarded.keys() - (miles_grok._SAMPLING | {"max_tokens"}))
        self.assertEqual(forwarded["max_tokens"], 64)
        self.assertEqual(forwarded["temperature"], 0.7)

    def test_original_tokens_logprobs_and_assistant_fields_survive(self):
        state = SessionStateV2()
        root = node(state, None, [1, 2, 3], [10, 11], _ORIGINAL + [_TOOL_CALL], [-0.125, -0.25])
        tool = {"role": "tool", "tool_call_id": "a", "content": "file content"}
        self.assertIs(state.tree.find_attach_point(root.path_messages() + [tool]).node, root)
        rewritten = {**_TOOL_CALL, "reasoning_content": "changed"}
        self.assertIsNone(state.tree.find_attach_point(_ORIGINAL + [rewritten, tool]).node)
        leaf = node(state, root, [1, 2, 3, 10, 11, 20, 21], [30, 31],
                    [tool, {"role": "assistant", "content": "done"}], [-0.5, -1.0])
        (sample,) = samples(state)
        self.assertEqual(sample.tokens, leaf.token_ids)
        self.assertEqual(sample.rollout_log_probs, [-0.125, -0.25, 0.0, 0.0, -0.5, -1.0])
        self.assertEqual(full_mask(sample), [0, 0, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(root.path_messages()[-1], _TOOL_CALL)

    def test_a_shared_completion_trains_in_one_branch(self):
        state = SessionStateV2()
        root = node(state, None, [1, 2, 3], [10, 11], _ORIGINAL + [_TOOL_CALL], [-0.125, -0.25])
        early = node(state, root, root.token_ids + [20], [30],
                     [{"role": "user", "content": "branch one"}, {"role": "assistant", "content": "first"}], [-0.5])
        later = node(state, root, root.token_ids + [21], [31],
                     [{"role": "user", "content": "branch two"}, {"role": "assistant", "content": "second"}], [-1.0])
        material = {sample.metadata["leaf"]["node_id"]: sample for sample in samples(state)}
        self.assertEqual(set(material), {early.seq, later.seq})
        self.assertEqual(full_mask(material[early.seq]), [0, 0, 0, 1, 1, 0, 1])
        self.assertEqual(full_mask(material[later.seq]), [0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(material[later.seq].rollout_log_probs[:2], [-0.125, -0.25])

    def test_compaction_preserves_both_roots_with_one_rollout_reward(self):
        state = SessionStateV2()
        root = node(state, None, [1, 2, 3], [10, 11], _ORIGINAL + [_TOOL_CALL], [-0.125, -0.25])
        summary = node(state, root, root.token_ids + [20, 21], [40, 41], [
            {"role": "tool", "tool_call_id": "a", "content": "file content"},
            {"role": "user", "content": "summary request"},
            {"role": "assistant", "content": "summary"},
        ], [-0.5, -1.0])
        compacted = _ORIGINAL + [{"role": "user", "content": "summary copied as context"}]
        self.assertIsNone(state.tree.find_attach_point(compacted).node)
        continued = node(state, None, [1, 2, 3, 40, 41, 22], [50, 51],
                         compacted + [{"role": "assistant", "content": "done"}], [-0.75, -1.25])
        self.assertEqual(len(state.tree.roots), 2)
        agent = {"rl_harness": {"status": "graded", "reward": 1.0}}
        material = samples(state, agent)
        self.assertEqual(len(material), 2)
        by_leaf = {sample.metadata["leaf"]["node_id"]: sample for sample in material}
        self.assertEqual(by_leaf[summary.seq].tokens, summary.token_ids)
        self.assertEqual(full_mask(by_leaf[summary.seq]), [0, 0, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(full_mask(by_leaf[continued.seq]), [0, 0, 0, 0, 0, 0, 1, 1])
        summary_training_count = sum(
            mask for sample in material for token, mask in zip(sample.tokens, full_mask(sample)) if token in {40, 41}
        )
        self.assertEqual(summary_training_count, 2)
        rewards = asyncio.run(miles_grok.reward(None, material))
        self.assertEqual(rewards, [1.0, 1.0])
        for sample, score in zip(material, rewards):
            sample.reward, sample.rollout_id, sample.group_index = score, 7, 0
        other_attempt = Sample(reward=0.0, rollout_id=8, group_index=0)
        args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=False)
        normalized = _normalize_rewards_by_rollout(args, material + [other_attempt], rewards + [0.0], None)
        self.assertEqual(normalized, [0.5, 0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
