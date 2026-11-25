"""Deterministic refund tools (agent PRD 6.4 - 6.7).

All eligibility rules, recharge selection and the approval limit live here, in code.
The LLM never decides eligibility (FR-29, FR-30, GR-05).
Only get_user_account and validate_refund are offered to the LLM; process_refund and
submit_to_support are called by application code only (FR-23a).
"""
import copy
import uuid

from refund_data import AccountStore

REFUND_WINDOW_DAYS = 7  # FR-25
APPROVAL_LIMIT = 499  # FR-36: at or below this, the app may execute after confirmation


def _fmt(amount: float) -> str:
    return f"₹{amount:,.0f}" if float(amount).is_integer() else f"₹{amount:,.2f}"


def get_user_account(store: AccountStore, user_id: str) -> dict:
    """FR-19: account and recharge details for the user."""
    user = store.get_user(user_id)
    if user is None:
        return {"found": False, "user_id": user_id, "reason": "No account found."}
    return {"found": True, "user_id": user_id, **copy.deepcopy(user)}


def recharge_age(recharge: dict) -> str:
    days = recharge["recharge_age_days"]
    return "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"


def recharge_phrase(recharge: dict) -> str:
    """Customer-facing description, e.g. 'your ₹499 recharge from yesterday' (no internal IDs)."""
    age = recharge_age(recharge)
    return f"your {_fmt(recharge['recharge_amount'])} recharge {'from ' if age != 'today' else ''}{age}"


def _rule_failure(recharge: dict, amount: float) -> str | None:
    """First failing rule for this recharge, or None if eligible (FR-24 .. FR-28)."""
    amt = _fmt(recharge["recharge_amount"])
    if recharge["recharge_status"] != "completed":
        return f"Your {amt} recharge isn't completed yet (status: {recharge['recharge_status']})."
    if recharge["recharge_age_days"] > REFUND_WINDOW_DAYS:
        return (f"Your {amt} recharge was made {recharge['recharge_age_days']} days ago, "
                f"outside the {REFUND_WINDOW_DAYS}-day refund window.")
    if recharge["refund_status"] == "refunded":
        return f"Your {amt} recharge has already been refunded."
    if amount > recharge["recharge_amount"]:
        return f"The requested amount {_fmt(amount)} exceeds the recharge amount {amt}."
    return None


def select_recharge(recharges: list[dict], amount: float) -> dict | None:
    """Deterministically pick the recharge a refund applies to (FR-27a).

    1. A recharge whose amount matches exactly identifies that recharge, even if it is
       already refunded (an unrefunded exact match is preferred when there are several).
    2. Otherwise, the smallest eligible recharge that covers the amount (most recent on ties).
    3. Otherwise, the smallest recharge that covers the amount, so validation can say why
       it fails; or, if none covers it, the largest recharge (amount exceeds it).
    """
    if not recharges:
        return None
    exact = [r for r in recharges if r["recharge_amount"] == amount]
    if exact:
        exact.sort(key=lambda r: (r["refund_status"] == "refunded", r["recharge_age_days"]))
        return exact[0]
    covering = sorted(
        (r for r in recharges if r["recharge_amount"] >= amount),
        key=lambda r: (r["recharge_amount"], r["recharge_age_days"]),
    )
    for r in covering:
        if _rule_failure(r, amount) is None:
            return r
    if covering:
        return covering[0]
    return max(recharges, key=lambda r: r["recharge_amount"])


def validate_refund(store: AccountStore, user_id: str, amount: float) -> dict:
    """FR-20: decide eligibility only. The approval limit is a separate step (6.7)."""
    result = {"eligible": False, "requested_amount": amount, "recharge_amount": None,
              "recharge_id": None}
    user = store.get_user(user_id)
    if user is None:
        return {**result, "reason": "No account found."}
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {**result, "reason": "The refund amount is not a number."}
    result["requested_amount"] = int(amount) if amount.is_integer() else amount
    if amount <= 0:  # FR-26
        return {**result, "reason": "The refund amount must be greater than zero."}

    recharge = select_recharge(user["recharges"], amount)
    if recharge is None:
        return {**result, "reason": "There is no recharge on this account to refund."}
    result.update(recharge_amount=recharge["recharge_amount"], recharge_id=recharge["recharge_id"])
    failure = _rule_failure(recharge, amount)
    if failure:
        return {**result, "reason": failure}
    return {**result, "eligible": True, "reason": "Refund is eligible."}


def eligible_recharges(store: AccountStore, user_id: str) -> list[dict]:
    """Recharges that could be refunded in full right now (used when no amount was given)."""
    user = store.get_user(user_id) or {"recharges": []}
    return [r for r in user["recharges"] if _rule_failure(r, r["recharge_amount"]) is None]


def process_refund(store: AccountStore, user_id: str, amount: float) -> dict:
    """FR-21, FR-23: SIMULATED refund. Called only by app code after validation + confirmation.

    Re-checks validation and the approval limit itself, so it can never refund an invalid
    or above-limit request even if called by mistake (GR-01, GR-08).
    """
    if float(amount) > APPROVAL_LIMIT:
        return {"success": False, "amount": amount, "status": "not_refunded",
                "reason": f"Refunds above {_fmt(APPROVAL_LIMIT)} cannot be executed by this application."}
    check = validate_refund(store, user_id, amount)
    if not check["eligible"]:
        return {"success": False, "amount": amount, "status": "not_refunded", "reason": check["reason"]}

    user = store.get_user(user_id)
    recharge = next(r for r in user["recharges"] if r["recharge_id"] == check["recharge_id"])
    recharge["refund_status"] = "refunded"  # FR-33; FR-28a: the whole recharge is consumed
    refund_id = f"ref_{uuid.uuid4().hex[:8]}"  # FR-34: simulated reference, no payment API (FR-35)
    store.refunds.append({"refund_id": refund_id, "user_id": user_id, "amount": check["requested_amount"],
                          "recharge_id": recharge["recharge_id"]})
    return {"success": True, "refund_id": refund_id, "amount": check["requested_amount"],
            "recharge_id": recharge["recharge_id"], "status": "refunded"}


def find_submission(store: AccountStore, user_id: str, recharge_id: str | None) -> dict | None:
    """An existing Support Team request for this recharge, if one was already submitted."""
    return next((s for s in store.support_submissions
                 if s["user_id"] == user_id and s["recharge_id"] == recharge_id), None)


def submit_to_support(store: AccountStore, user_id: str, amount: float, recharge_id: str | None = None) -> dict:
    """FR-21a, FR-38: hand an above-limit request to the Support Team. Refund status is unchanged.

    One request per recharge: asking again returns the existing request instead of a duplicate.
    The Support Team's review is outside this application and is never simulated (GR-09).
    """
    existing = find_submission(store, user_id, recharge_id)
    if existing:
        return {"submitted": False, "already_submitted": True, "submission_id": existing["submission_id"],
                "amount": existing["amount"], "recharge_id": recharge_id, "status": "submitted_for_review"}
    submission_id = f"sup_{uuid.uuid4().hex[:8]}"
    store.support_submissions.append({"submission_id": submission_id, "user_id": user_id,
                                      "amount": amount, "recharge_id": recharge_id})
    return {"submitted": True, "already_submitted": False, "submission_id": submission_id,
            "amount": amount, "recharge_id": recharge_id, "status": "submitted_for_review"}
