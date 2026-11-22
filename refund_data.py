"""Hard-coded demo account data (agent PRD FR-14 .. FR-18a). No real CRM or billing."""
import copy

# Treated as if it came from the authenticated session (FR-15, FR-16). Never asked of the user.
DEMO_USER_ID = "user_123"

USERS = {
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {
                "recharge_id": "rch_456",
                "recharge_amount": 999,
                "recharge_status": "completed",
                "recharge_age_days": 2,
                "refund_status": "not_refunded",
            },
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "not_refunded",
            },
        ],
    }
}


class AccountStore:
    """In-memory copy of USERS for one session, so simulated refunds don't leak between sessions."""

    def __init__(self, users: dict | None = None):
        self.users = copy.deepcopy(users if users is not None else USERS)
        self.refunds: list[dict] = []  # simulated refunds executed
        self.support_submissions: list[dict] = []  # requests handed to the Support Team

    def get_user(self, user_id: str) -> dict | None:
        return self.users.get(user_id)
