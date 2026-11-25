"""Command line for training (the Train tab will use the same calls).

    python -m narrator.training list                                 datasets ready, recipes, back ends, runs
    python -m narrator.training check-backend [kaggle]               signed in? GPU hours left?
    python -m narrator.training send alice-book=alice marcus-book=marcus --model narrators [--recipe orpheus]
                                     [--backend kaggle] [--start-from narrators --replay 25] [--epochs N] [--max-clips N] [--dry-run]
    python -m narrator.training status <model>
    python -m narrator.training finish <model> [--gender alice=female,marcus=male] [--name "alice=Alice,marcus=Marcus"]
"""
import sys

from . import BACKENDS, RECIPES, check, finish, ready_voices, runs, send


def _opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def _pairs(value):
    return dict(x.split("=", 1) for x in value.split(",") if "=" in x) if value else {}


def main(args):
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "list":
        for v, n, h in ready_voices():
            print(f"dataset  {v}: {n:,} clips, {h:.1f} h")
        for r in RECIPES.values():
            print(f"recipe   {r.id}: {r.name} [{r.readiness}]")
        for b in BACKENDS.values():
            print(f"backend  {b.id}: {b.name} [{b.readiness}] {b.pricing}")
        for m, r in runs().items():
            print(f"run      {m}: {r.get('recipe')} on {r.get('backend')}, {r.get('hours')} h, {r.get('state')} {r.get('url', '')}")
    elif cmd == "check-backend":
        print(BACKENDS[args[1] if len(args) > 1 else "kaggle"].check()["detail"])
    elif cmd == "send":
        flags = {"--model", "--recipe", "--backend", "--start-from", "--replay", "--epochs", "--max-clips"}
        specs = [a for i, a in enumerate(args[1:], 1) if not a.startswith("--") and args[i - 1] not in flags]
        model = _opt(args, "--model") or (specs[0].partition("=")[2] or specs[0]) if specs else None
        if not specs:
            sys.exit("Which voices? e.g.  send alice-book=alice --model alice")
        send(specs, model, recipe=_opt(args, "--recipe", "orpheus"), backend=_opt(args, "--backend", "kaggle"),
             dry_run="--dry-run" in args, start_from=_opt(args, "--start-from"),
             replay=float(_opt(args, "--replay", 25)) / 100, max_clips=int(_opt(args, "--max-clips", 0)),
             **({"epochs": float(_opt(args, "--epochs"))} if "--epochs" in args else {}))
    elif cmd == "status":
        st = check(args[1])
        print(f"{args[1]}: {st.state} {st.detail} {st.url}")
    elif cmd == "finish":
        print(finish(args[1], genders=_pairs(_opt(args, "--gender")), names=_pairs(_opt(args, "--name"))))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
