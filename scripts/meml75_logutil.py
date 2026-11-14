import sys
from contextlib import contextmanager

# NOTE (v5.1): module-level debug banner removed.
# It used to print "=== [MEML75_LOGUTIL v5] SUCCESSFULLY LOADED ==="
# on every import, polluting 4 log files
# (lepton_untested_v2, a2_weight_audit_v2,
#  reconcile_twisted_vs_weight0_v2, gvw_tau_closed_form_v2).
# The module is only ever imported, never run as __main__,
# so wrapping the banner in `if __name__ == "__main__"` would
# just make it dead code. Removed entirely.

@contextmanager
def safe_log(filename, **kwargs):
    try:
        f = open(filename, 'w', encoding='utf-8')
    except Exception:
        f = sys.stdout
        
    class Logger:
        def __call__(self, *args, **kw):
            print(*args, **kw)
            if f != sys.stdout:
                print(*args, **kw, file=f)
        def write(self, text):
            sys.stdout.write(text)
            if f != sys.stdout:
                f.write(text)
        def flush(self):
            sys.stdout.flush()
            if f != sys.stdout:
                f.flush()
        def info(self, msg, *args, **kw): self.write(str(msg) + "\n")
        def debug(self, msg, *args, **kw): self.write(str(msg) + "\n")
        def warning(self, msg, *args, **kw): self.write(str(msg) + "\n")
        def error(self, msg, *args, **kw): self.write(str(msg) + "\n")
            
    logger = Logger()
    logger.path = filename
    try:
        yield logger
    finally:
        if f != sys.stdout:
            f.close()