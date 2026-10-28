Local Harbor integration check

Use Harbor 0.25, Docker, host.docker.internal resolving to the host, and the
bundled Linux runtime in ../runtime. The runtime and Docker worker must use the
same CPU architecture. From the repository root:

  python checks/check_harbor.py

The selected Python must have harbor/requirements.txt installed. Optional
--runtime-dir and --output-dir select the bundle and a fresh evidence directory.
By default, evidence is retained under the operating system's temporary folder.

This runs the actual Harbor AgentFactory, Trial, Docker environment, adapter,
Rust tools, compaction, and Harbor verifier against a scripted local HTTP server.
The sandbox calls that server directly through host.docker.internal. This
address belongs to the local fixture, not the agent adapter. The server listens
on the host's network interfaces and accepts only a fixed fake API key. No paid
model endpoint is used. A pinned multi-architecture Debian image supplies the
OS. Docker may download that image if it is not already cached.

The completed trial exercises all eight tools, forces one compaction, checks
that the todo and live background task survive it, then verifies the edited
file and cleanup of both explicitly stopped and leftover background processes.
The second trial has an eight-second Harbor timeout and a longer Rust deadline;
it also overwrites the old PID-file location. The verifier requires its
background process to be gone before verification. The third trial forces a
missing host process identity and checks that Harbor skips verification and
tears down the environment.
The fourth trial kills the Rust runner while its tool process keeps running.
Harbor must skip verification and destroy the environment.
The verifier shares the task's process namespace so it can test that cleanup.

A passing reward belongs only to this integration fixture. It is not a
Terminal-Bench score or evidence of model task-solving ability.
