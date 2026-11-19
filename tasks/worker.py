"""Executed only inside the OS sandbox. Expected answers never enter this process."""
import json
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
payload = json.load(sys.stdin)
namespace = {"__name__": "candidate"}
exec(compile(payload["code"], "<candidate>", "exec"), namespace)
results = [namespace["solve"](*args) for args in payload["inputs"]]
print(json.dumps(results, allow_nan=False))
