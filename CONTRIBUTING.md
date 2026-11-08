# Contributing

MEDEA uses the **GitHub flow** repository workflow:

- All features and changes are developed on their own branch (created from `dev`).
  This way all development can be viewed by other collaborators.
- When your feature is ready for review, create a Pull Request in GitHub to merge your changes into `dev`.

For more information on the GitFlow strategy, see [its documentation](https://docs.github.com/en/get-started/using-github/github-flow) on GitHub.

Prepare a development checkout with the following steps:

```
git clone git@github.com:CARRE-PSAAP/MEDEA.git
git switch dev
git switch -c <your-branch-name>

...

git push
```

Make sure to regularly pull the latest changes from dev:

```
git switch dev
git pull origin dev
git switch <your-branch-name>
git merge dev
```

Make sure to `git push` on occasion so your work can be seen by the other developers.
When your feature is ready to review, create a pull request on the GitHub web interface, or

```
gh pr create
```

then follow the prompts.

## Code Format and Style

Python code uses the `black` style. Run `black .` from project root to format your code before commiting.

C++ code uses the `Google` style, and uses `clang-format` to perform the formatting. Run `clang-format -i <file.h> <file.cpp>` to format your code before commiting.

## Versioning

MEDEA uses [semantic versioning](https://semver.org/#semantic-versioning-200). When changes are merged into `main` from `dev`, this increments the code version, which is numbered MAJOR.MINOR.PATCH:

- MAJOR version: "breaking" API changes
- MINOR version: backward-compatible additions to functionality
- PATCH version: backward-compatible bug fixes

MEDEA will be on major version zero (0.x.y) until a stable API is developed, ready for interfacing with MC/DC.

