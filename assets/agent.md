---
name: remanie-train
description: Grok Build's grok-build-concise agent with the eight trial tools.
toolConfig:
  tools:
    - id: GrokBuildConcise:run_terminal_cmd
    - id: GrokBuildConcise:read_file
    - id: GrokBuildConcise:search_replace
    - id: GrokBuild:list_dir
    - id: GrokBuild:grep
    - id: GrokBuild:kill_terminal_command
    - id: GrokBuild:todo_write
    - id: GrokBuild:get_terminal_command_output
injectDefaultTools: false
agentsMd: false
discoverSkills: false
inheritSkills: false
---
