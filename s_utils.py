import subprocess
import errno
import sys


def run_command(cmds, args, cwd=None, verbose=False, hide_stderr=False):
    assert isinstance(cmds, list)
    print(cmds, args)
    p = None
    for c in cmds:
        try:
            # remember shell=False, so use git.cmd on windows, not just git
            p = subprocess.Popen([c] + args, cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=(subprocess.PIPE if hide_stderr
                                         else None))
            break
        except EnvironmentError as ex:
            print(os.getcwd(),ex)
            e = sys.exc_info()[1]
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % args[0])
                print(e)
            return False, str(e)
    else:
        if verbose:
            print("unable to find command, tried %s" % (cmds,))
        return False, "no commands"
    stdout = p.communicate()[0].strip()
    if sys.version >= '3':
        stdout = stdout.decode()
    if p.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % args[0])
        return False, stdout
    return True, stdout

