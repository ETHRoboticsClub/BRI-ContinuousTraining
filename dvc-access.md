# Data access with DVC

Datasets and trained models are not stored in git. They live in the S3 bucket `hri-control` (region `us-east-1`)
and are versioned with [DVC](https://dvc.org). Git only stores small pointer files (`*.dvc`, `dvc.lock`) that say
which version of the data goes with each commit.

## What you receive

Marcos creates an AWS user for you named `hri-<yourname>` and sends you an **access key** in a private message:

- `Access key ID` (starts with `AKIA...`)
- `Secret access key`

This key only works from the command line. It does not give you a login to the AWS web console, and you don't need one.

Keep the key private: never commit it, paste it in a channel, or put it in code. Once it is configured (step 2),
delete the message that contained it. If you think it leaked, tell Marcos so he can revoke it.

## 1. Install

You need Python ≥ 3.10, git and the AWS CLI v2.

```bash
# AWS CLI v2 (Linux; macOS/Windows: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# the project, which installs DVC with S3 support
git clone <repo-url> && cd HRI-movement-decoder
pip install -e ".[dev]"
```

## 2. Configure your key

The profile **must** be called `hri`: the repo's DVC config uses that name.

```bash
aws configure --profile hri
```

| Prompt                | Answer                    |
|-----------------------|---------------------------|
| AWS Access Key ID     | your key ID (`AKIA...`)   |
| AWS Secret Access Key | your secret               |
| Default region name   | `us-east-1`               |
| Default output format | `json`                    |

The key is stored in `~/.aws/credentials` on your machine, outside the repo.

Check that it works:

```bash
aws sts get-caller-identity --profile hri   # should show user/hri-<yourname>
aws s3 ls s3://hri-control --profile hri    # should list the bucket (no error)
```

## 3. Get the data

```bash
dvc pull
```

## Daily workflow

**Get the latest code and data**

```bash
git pull
dvc pull
```

**Add a new dataset**

```bash
# put the files in data/raw/<name>/ first
dvc add data/raw/<name>
git add data/raw/<name>.dvc data/raw/.gitignore
git commit -m "Add <name>"
dvc push        # upload the data
git push        # share the pointer
```

**Run the pipeline and share the results**

```bash
dvc repro                   # rebuild dataset + train, only what changed
git add dvc.lock params.yaml
git commit -m "Train with ..."
dvc push
git push
```

### Rules

1. **Always `dvc push` before `git push`.** If you push the git commit without the data, everyone else's `dvc pull`
   fails with "missing cache files".
2. **`dvc add` new data before any `git add`.** Raw data folders are only ignored by git after `dvc add` has run on them,
   so `git add .` before that would commit the data into git.
3. Optional: run `dvc install` once in the repo. It adds git hooks so `git push` also runs `dvc push`, and
   `git checkout` also updates the data.

## Training on a GPU (SkyPilot)

`sky/train.yaml` runs the same pipeline (`dvc pull` → `dvc repro` → `dvc push`) on an AWS A10G machine.
It copies your `~/.aws/credentials` to the machine, so step 2 above must be done first.

```bash
sky launch -c hri-gpu-<yourname> sky/train.yaml --env SMOKE_TEST=1   # once: create the machine + check GPU/S3
sky exec hri-gpu-<yourname> sky/train.yaml                           # each run: syncs your local code, trains
sky/fetch.sh hri-gpu-<yourname>                                      # brings dvc.lock + metrics back, pulls the model
git add dvc.lock && git commit -m "..." && git push
```

The machine stops by itself after 30 idle minutes (`sky start hri-gpu-<yourname>` to resume, packages stay installed).
It costs about $1/h while running. Use `sky down hri-gpu-<yourname>` when you won't need it for a while.
Change parameters by editing `params.yaml` locally before `sky exec`.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Unable to locate credentials` / `NoCredentials` | Step 2 not done, or the profile isn't named `hri`. Check with `aws configure list --profile hri`. |
| `AccessDenied` on `hri-control` | Your user isn't in the `hri-user` group yet, so ask Marcos. (`s3 ls` with no bucket name is always denied; that's expected.) |
| `InvalidAccessKeyId` / `SignatureDoesNotMatch` | Key or secret mistyped. Run `aws configure --profile hri` again. |
| `dvc pull`: `No file hash info found for data/processed/...` or `runs/...` | The pipeline outputs haven't been pushed for this commit yet. Run `dvc repro` yourself, or ask whoever trained it to `dvc push`. |
| `dvc pull`: `missing cache files` for other paths | Someone pushed git without `dvc push`. Ask them to run `dvc push`. |
| I want to use a different AWS profile name | `dvc remote modify --local storage profile <name>` (stays on your machine and isn't committed). |
