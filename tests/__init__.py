"""Full-loop integration tests (Trajectory → Controller → Estimator → Plant → Engine).

These are the LeKiwi demo's acceptance tests: a full closed loop against MuJoCo
with the bundled assets. They require the ``[mujoco]`` extra and run by default
via ``make test``.
"""
