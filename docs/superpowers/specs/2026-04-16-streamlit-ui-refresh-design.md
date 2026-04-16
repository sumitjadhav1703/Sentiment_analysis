# Streamlit UI Refresh Design

**Date:** 2026-04-16

**Goal:** Improve the existing Streamlit app’s front end into a clean professional dashboard while keeping the current single-text prediction behavior unchanged.

## Scope

This design covers:
- A visual refresh of `app.py`
- Better layout, spacing, hierarchy, and styling
- Clearer presentation of prediction input and result
- Supporting information panels below the main interaction

This design does not cover:
- New model behavior
- Batch prediction
- Multi-page navigation
- Authentication
- New backend services

## Current State

The current Streamlit app already supports:
- A single text input
- Validation for blank input
- Prediction through `predict.py`
- Display of predicted label and confidence

The current limitation is presentation. The interface is functional but visually plain.

## Selected Direction

Use a clean professional dashboard layout with a strong single-column prediction flow.

Why this direction:
- The app currently performs one primary user action, so the UI should emphasize that path
- A single-column layout is clearer and more robust on both desktop and mobile
- A restrained dashboard treatment looks more professional than a generic form page

## Layout

### Header

The top of the page should include:
- A concise project label
- A clear title
- A one- or two-line summary describing the app

This section should establish the app as a polished NLP project rather than a raw prototype.

### Main Prediction Card

The primary interaction should sit in a prominent card near the top of the page.

It should contain:
- The text input area
- The predict button
- Supporting helper text if needed

The card should feel deliberate and centered, with strong spacing and clear visual separation from the page background.

### Result Card

Prediction output should appear directly below the main prediction card.

It should highlight:
- Predicted emotion label
- Confidence score

The result should be easier to scan than the current output, with stronger hierarchy and cleaner formatting.

### Supporting Panels

Below the main prediction flow, add supporting informational sections such as:
- About the model
- How to use

These panels should add credibility and context without competing with the prediction flow.

## Visual Style

The style should feel:
- Professional
- Minimal
- Structured
- Calm rather than flashy

Guidance:
- Use a neutral or subdued palette
- Prefer subtle borders, shadows, and spacing over loud color blocks
- Maintain strong typography and hierarchy
- Avoid playful or overly decorative effects

## Responsive Behavior

The refreshed layout must work on both desktop and mobile:
- Main interaction remains readable on smaller screens
- Supporting panels should stack naturally
- No side-by-side dependence for critical content

## Behavior Constraints

The refresh must not change the current app logic:
- Keep single-text prediction only
- Keep blank-input validation
- Keep label + confidence result
- Keep missing-artifact error handling

This is a presentation upgrade, not a behavior rewrite.

## Testing Strategy

The implementation should preserve the existing helper-focused tests where possible.

Additional testing can remain light as long as:
- The helper tests continue to pass
- The app still loads
- The prediction path still works with the saved model artifact

## Success Criteria

The UI refresh is successful when:
- The app looks intentionally designed rather than default Streamlit
- The prediction flow is still the most prominent part of the page
- Result presentation is clearer than before
- The layout works on desktop and mobile
- Existing functionality remains intact
