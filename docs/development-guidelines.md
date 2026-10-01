# Development guidelines

## Branches

Use `main` for integrated, reviewed work. Each member develops on one long-lived branch:

```text
main
├── member1-object-detection
├── member2-person-tracking
├── member3-behaviour-analysis
└── member4-risk-prediction
```

| Branch | Owner | Working folder |
| --- | --- | --- |
| `member1-object-detection` | Member 1 | `member1-object-detection/` |
| `member2-person-tracking` | Member 2 | `member2-person-tracking/` |
| `member3-behaviour-analysis` | Member 3 | `member3-behaviour-analysis/` |
| `member4-risk-prediction` | Member 4 | `member4-risk-prediction/` |

Create your branch from an up-to-date `main`:

```bash
git checkout main
git pull
git checkout -b member1-object-detection
```

Replace the branch name with your own.

## Daily workflow

1. Work on your own branch.
2. Commit regularly, with messages that say what changed and why.
3. Push your branch to GitHub.
4. Open a pull request into `main` when a piece of work is ready for the team to see.
5. Merge only after another member has looked at the pull request.

```bash
git status
git add member1-object-detection
git commit -m "Describe the change and why it was made."
git push -u origin member1-object-detection
```

Keep each commit inside your own module when you can. A pull request that only touches your folder is easier to review.

## Ownership rules

- Commit your own work to your own module folder.
- You may read every folder. The repository is shared so the team can see each other's work.
- Do not directly overwrite another member's module. If you find a problem there, tell that member or open a pull request they can review.
- Do not reformat or move another member's files as a side effect of your change.
- Shared files are `README.md`, `requirements.txt`, `.gitignore`, `.env.example`, `docs/`, and later `integration/`. Change those in a pull request and ask the team to review.

## Shared interfaces

The JSON contracts in [api-contract.md](api-contract.md) are shared.

- Discuss a contract change before editing the file.
- Update the contract in the same pull request that needs the new fields, or in an earlier pull request that lands first.
- Do not privately change field names inside your module and expect the next member to guess.

`integration/` is for connecting the four modules after their outputs match the contract. Do not put a member's private implementation there.

## What not to commit

Do not commit any of the following unless the team has specifically agreed that the file is required in Git:

- Datasets and raw or processed video (`data/raw/`, `data/processed/`, `data/samples/`)
- Large model files (`.pt`, `.pth`, `.onnx`, and similar weights under `models/`)
- Generated outputs (`outputs/`)
- Result videos, screenshots, and bulky reports (`results/`)
- Secrets (`.env`, keys, tokens, passwords)

`.gitignore` already excludes these. `.env.example` is the committed template. Copy it to `.env` locally.

If a small file of an ignored type really must be stored in Git, agree first, then add that one file with `git add -f`.

## Dependencies

`requirements.txt` is empty of packages on purpose. When implementation starts, add the packages your module needs, with versions, under a short comment naming the module. Do not install unstated packages into the repository files without listing them.

## Local setup

From the repository root, after implementation begins:

```bash
python -m venv .venv
```

Activate the virtual environment, then install from `requirements.txt` once it lists real packages. There is nothing to install at the current stage.

Copy environment settings only when a module needs them:

```bash
copy .env.example .env
```

On macOS or Linux, use `cp .env.example .env`.

## Pull request checklist

- The change stays inside the owner's module, or the shared-file edit was discussed.
- No datasets, weights, videos, outputs, or `.env` files are included.
- Interface changes are reflected in `docs/api-contract.md`.
- The README inside the module still describes the input and output correctly.
