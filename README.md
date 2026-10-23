# P08 — SciPy regression replay

**Field:** Software verification  
**Size:** Medium (one project season)  
**Time:** 6-8 weeks

## Question
If you replay the same numerical inputs across 10 years of SciPy releases, where did answers silently change?

## Why it matters
Your VODE regression appeared silently in 1.17.0. A systematic replay could find others nobody noticed.

## Builds on
SciPy issue #26384

## Method
1. Install releases 1.0-1.17 in isolated environments (conda/uv)
2. Collect inputs from SciPy's own tests + your reference cases
3. Diff outputs between consecutive releases; triage changes beyond tolerance
4. Classify: bug fixed, bug introduced, intended change

## Deliverable
Regression map + upstream issues.

## Risks
Old releases need old Pythons, so use containers and keep memory low.

## Before you start: novelty check
Search Google Scholar / arXiv / GitHub for: _numerical regression across versions scientific library_. If the question is already answered, narrow it or pick another project.

## Possible outlet
Workshop paper; Examen Silentii

## Log
_(dates, decisions, results)_
