# 0007. A question asked without a user sees nothing

- **Status:** Accepted (implementation pending: PRD requirement R7)
- **Date:** 2026-10-06

## Context
After permission filtering ([0006](0006-enforce-permissions-in-retrieval-filter.md)) is added, existing callers of the question function don't pass a user. We had to decide what those calls should see.

## Decision
A question without a user is treated like an unknown user: no groups, so no chunks are retrieved and the reply is "not found".

## Technical approach
- The question function builds the permission filter from the user's groups. With no user, the group list is empty, so nothing matches.
- No code path skips the permission filter.

## Alternatives considered
- **No user means see everything (backward compatible):** rejected. Any caller could bypass permissions just by leaving the user out.
- **Raise an error when the user is missing:** louder for developers, but treating the missing user as an empty group list keeps a single code path and the same reply as any other denied request.

## Tradeoffs
- **Gained:** forgetting to pass a user fails safe instead of leaking restricted content.
- **Gave up:** existing callers, including the demo, must be updated to pass a user, or they get "not found" for everything.
