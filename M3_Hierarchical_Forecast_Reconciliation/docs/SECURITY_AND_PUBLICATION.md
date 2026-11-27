# GitHub privacy and data publication notes

## What is deliberately excluded from the repository

- Raw M5 data and price files.
- The full prediction Parquet tree, about 901 MB.
- Python caches, compiled bytecode and editor scratch files.
- Names, student EIDs, private email addresses, local usernames and absolute paths.
- Model weights, login credentials and environment secrets.

## Before the first commit

Git writes the commit author into the commit history. If a personal machine is configured with a private email address, switch to the GitHub noreply address first:

```bash
git config user.name "Your GitHub Name"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
```

This must be done before the first commit. Changing the configuration afterwards does not remove the author information from earlier commits.

## Recommended publication approach

1. Keep only code, small contract files and diagnostic summaries in Git.
2. Write the full predictions to a local directory listed in `.gitignore`, for example `prediction_outputs/`.
3. To share the full results, use a controlled attachment or a GitHub Release, and verify the data licence first.
4. When merging into the team repository, also check whether the other module documents contain member names or EIDs.

## Pre-publication scan

Run `gitleaks` or GitHub secret scanning and review code, Markdown, JSON and commit author information by hand. Then confirm the repository contains no Python caches, operating-system metadata, personal directories or private email addresses.

The public package keeps only role descriptions such as "Module 3 lead" and no specific student identity.


