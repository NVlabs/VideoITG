# VideoITG Contribution Guidelines

Thank you for your interest in contributing to VideoITG! To ensure efficient and standardized collaboration, please follow the guidelines below.

---

## Issue Tracking

- All feature requests, bug fixes, or change proposals should begin with an [Issue](https://github.com/NVlabs/VideoITG/issues).
- Issues should be confirmed and assigned by project maintainers before code development and submission.

---

## Coding Guidelines

- When adding or modifying code, please follow the existing conventions in the relevant file, submodule, module, and the overall project.
- To maintain consistent code formatting and style, please use tools such as `black`, `isort`, and `flake8` before submitting your code.
- Formatting example:
  ```bash
  # Format all Python files
  black .
  isort .
  flake8 .
  ```
- Avoid introducing unnecessary complexity to preserve maintainability and readability.
- Keep pull requests (PRs) as concise as possible:
  - Do not commit large blocks of commented-out code.
  - Each PR should address a single concern. If multiple unrelated issues need to be fixed, open separate PRs and indicate dependencies in the description.
- Write commit messages in imperative mood and follow [these rules](https://chris.beams.io/posts/git-commit/).
  - Recommended format:
    ```
    #<Issue Number> - <Commit Title>

    <Commit Body>
    ```
- Ensure the build log is clean, with no warnings or errors.
- Make sure all tests pass before submitting your code.
- All new features or modules must be accompanied by relevant documentation (e.g., README) and test cases.

---

## Pull Request Workflow

1. **Fork the Repository**  
   Fork the [VideoITG main repository](https://github.com/NVlabs/VideoITG).

2. **Local Development**  
   Clone your fork, switch to the target branch, and make your changes.
   ```bash
   git clone https://github.com/NVlabs/VideoITG.git
   git checkout <feature-branch>
   # Make changes and commit
   git push -u origin <feature-branch>
   ```

3. **Submit a PR**  
   - Once your changes are ready, submit a PR from your fork’s branch to the main repository’s target branch.
   - If your PR is a work in progress, prefix the title with `[WIP]`. Remove it when ready for review.
   - At least one maintainer must review your code.
   - Ensure all tests pass before merging.

---

## Signing Your Work

- All contributors must sign off on their commits using the `--signoff` (or `-s`) option, certifying that you have the right to submit the code and agree to the open-source license.
  ```bash
  git commit -s -m "Add new feature"
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- For details, see the [Developer Certificate of Origin 1.1](https://developercertificate.org/).

---

## Additional Notes

- When adding or changing features, use configuration files or command-line options to control optional functionality, rather than hardcoding.
- All new modules, plugins, etc., must include test cases and documentation.
- Thank you for your patience and contributions! Feedback and suggestions are always welcome.

---

If you have any questions, please open an Issue or contact the project maintainers.

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
