"""Keep this versioned Decoder entrypoint outside the candidate being repaired.

It uses change_sessions and the ordinary retained native controller. It never
imports the candidate controller in its coordinator and offers no science run.
Keep the prior saved package available when updating recovery itself.
"""
from .change_sessions import main

if __name__ == '__main__':
    raise SystemExit(main())
