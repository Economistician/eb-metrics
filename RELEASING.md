
# Release Process for `eb-metrics`

A guide for managing and automating the release process of `eb-metrics` to PyPI, including versioning, release triggers, and post-release validation.

---

## Overview
This document outlines the process for releasing a new version of `eb-metrics` to PyPI, including how to trigger the release, versioning guidelines, and post-release validation.

---

## Versioning
We follow **Semantic Versioning** (MAJOR.MINOR.PATCH) for versioning our packages. When making changes to `eb-metrics`, please update the version as follows:
- **Patch version**: for bug fixes or small improvements.
- **Minor version**: for new features that are backward-compatible.
- **Major version**: for breaking changes.

### Versioning Example:
- From `1.0.0` → `1.1.0` (minor update with new features).
- From `1.1.0` → `1.1.1` (patch update with bug fixes).

---

## Triggering a Release
Releases are triggered by creating a **version tag** in the repository.

### Steps:
1. **Create a new version tag**:
   ```bash
   git tag v<new_version> -m "Release <new_version>"
   git push origin v<new_version>
   ```
2. This will automatically trigger the **`pypi-release.yml`** workflow, which will handle the build and publishing process.

---

## Pre-release Checklist
Before triggering a release, please ensure the following:
- [ ] All tests pass.
- [ ] Version has been bumped according to [Semantic Versioning](#versioning).
- [ ] Changelog is updated with new features, fixes, or breaking changes.
- [ ] Documentation is up to date.

---

## Automated Workflow
The **`pypi-release.yml`** workflow performs the following steps:
1. **Build** the source distribution and wheel files.
2. **Publish** the package to **PyPI**.
3. After publishing, the **`pypi-smoke.yml`** workflow will be triggered to verify the release by installing it from PyPI and running basic checks.

---

## Post-release
Once the release is pushed to PyPI, the **`pypi-smoke.yml`** workflow will:
- Install the package from PyPI.
- Run basic import and functionality checks to ensure the release works as expected.

---

## Emergency Procedures
If an issue arises after the release:
1. Verify the issue in the **`pypi-smoke.yml`** logs.
2. If necessary, create a new patch version and re-release it.
3. Tag the new version and push it, triggering the release process again.

---

## Rollback Process
If a critical issue is found in a release after it's been pushed to PyPI:
1. Identify the issue and confirm it with the **`pypi-smoke.yml`** logs.
2. Create a hotfix by bumping the patch version (e.g., `1.0.0` → `1.0.1`).
3. Push the hotfix version and tag it.
4. Trigger the release pipeline and monitor the post-release smoke tests.

---

## Links
- [PyPI Guidelines](https://pypi.org/)
- [GitHub Actions Workflow Documentation](https://docs.github.com/en/actions)