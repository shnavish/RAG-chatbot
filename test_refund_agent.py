"""Deterministic tests for the refund rules and agent state machine (no LLM, no network).

Run:  .venv\\Scripts\\python -m unittest test_refund_agent -v
"""
import copy
import unittest
from unittest import mock

import refund_agent
from refund_agent import RefundAgent, RefundState, parse_confirmation
from refund_data import USERS, AccountStore
from refund_tools import process_refund, select_recharge, validate_refund

UID = "user_123"


def fake_planner(amount=None):
    """Stands in for the LLM: optionally 'calls' validate_refund with `amount`."""
    def plan(message, history):
        calls = [{"name": "get_user_account", "args": {"user_id": UID}, "result": {}}]
        if amount is not None:
            calls.append({"name": "validate_refund", "args": {"user_id": UID, "amount": amount}, "result": {}})
        return calls
    return plan


def store_with(**changes):
    users = copy.deepcopy(USERS)
    for recharge_id, fields in changes.items():
        for r in users[UID]["recharges"]:
            if r["recharge_id"] == recharge_id:
                r.update(fields)
    return AccountStore(users)


def make_agent(store, amount=None):
    return RefundAgent(store, UID, planner=fake_planner(amount))


class ValidationRules(unittest.TestCase):
    def test_eligible_499(self):
        r = validate_refund(AccountStore(), UID, 499)
        self.assertEqual(r, {"eligible": True, "reason": "Refund is eligible.", "requested_amount": 499,
                             "recharge_amount": 499, "recharge_id": "rch_457"})

    def test_exceeds_recharge(self):  # Scenario 2
        r = validate_refund(AccountStore(), UID, 2000)
        self.assertFalse(r["eligible"])
        self.assertIn("exceeds the recharge amount", r["reason"])

    def test_too_old(self):  # Scenario 3
        r = validate_refund(store_with(rch_457={"recharge_age_days": 15}), UID, 499)
        self.assertFalse(r["eligible"])
        self.assertIn("7-day refund window", r["reason"])

    def test_already_refunded(self):  # Scenario 4 / FR-27a exact match even if refunded
        r = validate_refund(store_with(rch_457={"refund_status": "refunded"}), UID, 499)
        self.assertFalse(r["eligible"])
        self.assertEqual(r["recharge_id"], "rch_457")
        self.assertIn("already been refunded", r["reason"])

    def test_not_completed(self):
        r = validate_refund(store_with(rch_457={"recharge_status": "pending"}), UID, 499)
        self.assertFalse(r["eligible"])

    def test_zero_and_negative(self):
        self.assertFalse(validate_refund(AccountStore(), UID, 0)["eligible"])
        self.assertFalse(validate_refund(AccountStore(), UID, -5)["eligible"])

    def test_partial_amount_selects_smallest_covering(self):
        self.assertEqual(select_recharge(USERS[UID]["recharges"], 300)["recharge_id"], "rch_457")
        self.assertEqual(select_recharge(USERS[UID]["recharges"], 600)["recharge_id"], "rch_456")

    def test_process_refund_refuses_above_limit_and_invalid(self):
        store = AccountStore()
        self.assertFalse(process_refund(store, UID, 999)["success"])
        self.assertFalse(process_refund(store, UID, 2000)["success"])
        self.assertEqual(store.refunds, [])


class Confirmation(unittest.TestCase):
    def test_parse(self):
        for yes in ["Yes, proceed", "yes", "Confirm", "Go ahead", "sure, go ahead", "yes please", "Yep"]:
            self.assertIs(parse_confirmation(yes), True, yes)
        for no in ["No", "Cancel", "don't proceed", "no thanks", "never mind", "I don't want to proceed"]:
            self.assertIs(parse_confirmation(no), False, no)
        for unclear in ["maybe", "ok", "ok thanks", "sure?", "yes but not now", "I want a refund of ₹499.",
                        "yes no", "₹499"]:
            self.assertIsNone(parse_confirmation(unclear), unclear)


class AgentFlows(unittest.TestCase):
    def test_scenario1_confirm_then_process(self):
        store = AccountStore()
        agent = make_agent(store, 499)
        with mock.patch.object(refund_agent, "process_refund", wraps=process_refund) as spy:
            reply = agent.handle("I want a refund of ₹499.")
            self.assertTrue(reply.needs_confirmation)
            self.assertEqual(agent.state, RefundState.AWAITING_CONFIRMATION)
            spy.assert_not_called()  # FR-36b: the request itself is not confirmation
            reply = agent.handle("Yes, proceed")
            spy.assert_called_once()
        self.assertEqual(agent.state, RefundState.REFUND_PROCESSED)
        self.assertIn("ref_", reply.text)
        self.assertEqual(store.get_user(UID)["recharges"][1]["refund_status"], "refunded")

    def test_ambiguous_then_cancel(self):
        store = AccountStore()
        agent = make_agent(store, 499)
        agent.handle("I want a refund of ₹499.")
        reply = agent.handle("maybe")
        self.assertTrue(reply.needs_confirmation)
        self.assertEqual(agent.state, RefundState.AWAITING_CONFIRMATION)
        reply = agent.handle("Cancel")
        self.assertEqual(agent.state, RefundState.REFUND_CANCELLED)
        self.assertEqual(store.refunds, [])

    def test_scenario2_invalid_never_processes(self):
        agent = make_agent(AccountStore(), 2000)
        with mock.patch.object(refund_agent, "process_refund") as spy:
            reply = agent.handle("Give me a refund of ₹2,000.")
            spy.assert_not_called()
        self.assertEqual(agent.state, RefundState.REFUND_REJECTED)
        self.assertIn("exceeds", reply.text)

    def test_scenario6_above_limit_submitted(self):
        store = AccountStore()
        agent = make_agent(store, 999)
        with mock.patch.object(refund_agent, "process_refund") as spy:
            reply = agent.handle("I want a refund of ₹999.")
            spy.assert_not_called()
        self.assertEqual(agent.state, RefundState.REFUND_SUBMITTED_FOR_REVIEW)
        self.assertFalse(reply.needs_confirmation)
        self.assertIn("submitted to our Support Team", reply.text)
        for word in ("approved", "rejected", "processed your", "has been refunded"):
            self.assertNotIn(word, reply.text.lower())  # FR-40a / GR-09
        self.assertEqual(store.get_user(UID)["recharges"][0]["refund_status"], "not_refunded")
        self.assertEqual(len(store.support_submissions), 1)

    def test_refunding_one_leaves_other_refundable_and_no_double_refund(self):
        store = AccountStore()
        agent = make_agent(store, 499)
        agent.handle("I want a refund of ₹499.")
        agent.handle("Yes, proceed")
        again = make_agent(store, 499).handle("Refund ₹499 again please")
        self.assertIn("already been refunded", again.text)
        other = make_agent(store, 999).handle("I want a refund of ₹999.")
        self.assertIn("submitted to our Support Team", other.text)

    def test_missing_amount_two_eligible_asks_which(self):
        agent = make_agent(AccountStore())
        reply = agent.handle("I want a refund.")
        self.assertEqual(agent.state, RefundState.AWAITING_AMOUNT)
        self.assertIn("₹999", reply.text)
        self.assertIn("₹499", reply.text)
        self.assertEqual(reply.choices, ["₹999", "₹499"])  # quick-reply buttons
        agent._planner = fake_planner(499)
        reply = agent.handle("₹499")  # what tapping the ₹499 button sends
        self.assertTrue(reply.needs_confirmation)

    def test_duplicate_support_request_is_not_resubmitted(self):
        store = AccountStore()
        first = make_agent(store, 999).handle("I want a refund of ₹999.")
        second = make_agent(store, 999).handle("I want a refund of ₹999.")
        self.assertEqual(len(store.support_submissions), 1)
        ref = store.support_submissions[0]["submission_id"]
        self.assertIn(ref, first.text)
        self.assertIn("already with our Support Team", second.text)
        self.assertIn(ref, second.text)

    def test_recharge_under_review_is_not_offered_again(self):
        store = AccountStore()
        make_agent(store, 999).handle("I want a refund of ₹999.")
        reply = make_agent(store).handle("I want a refund.")
        self.assertTrue(reply.needs_confirmation)  # only the ₹499 recharge is left to offer
        self.assertIn("₹499", reply.text)

    def test_ambiguous_reply_keeps_refund_pending_with_clear_message(self):
        agent = make_agent(AccountStore(), 499)
        agent.handle("I want a refund of ₹499.")
        reply = agent.handle("ok thanks")
        self.assertTrue(reply.needs_confirmation)
        self.assertIn("nothing has been refunded yet", reply.text)
        self.assertEqual(agent.state, RefundState.AWAITING_CONFIRMATION)

    def test_customer_messages_have_no_internal_recharge_ids(self):
        store = AccountStore()
        texts = [make_agent(store, 499).handle("I want a refund of ₹499.").text,
                 make_agent(store_with(rch_457={"recharge_age_days": 15}), 499).handle("refund ₹499").text,
                 make_agent(store_with(rch_457={"refund_status": "refunded"}), 499).handle("refund ₹499").text]
        agent = make_agent(AccountStore(), 499)
        agent.handle("I want a refund of ₹499.")
        texts.append(agent.handle("Yes, proceed").text)
        for t in texts:
            self.assertNotIn("rch_", t, t)

    def test_missing_amount_single_eligible_offers_it(self):
        agent = make_agent(store_with(rch_456={"refund_status": "refunded"}))
        reply = agent.handle("I want a refund.")
        self.assertTrue(reply.needs_confirmation)
        self.assertIn("₹499", reply.text)

    def test_missing_amount_single_above_limit_offer(self):
        store = store_with(rch_457={"refund_status": "refunded"})
        agent = make_agent(store)
        reply = agent.handle("I want a refund.")
        self.assertEqual(agent.state, RefundState.AWAITING_AMOUNT)
        self.assertIn("₹999", reply.text)
        reply = agent.handle("Yes")
        self.assertEqual(agent.state, RefundState.REFUND_SUBMITTED_FOR_REVIEW)

    def test_model_cannot_invent_amount(self):  # FR-44
        agent = make_agent(AccountStore(), 499)  # model "invents" 499; customer said no amount
        reply = agent.handle("I want a refund.")
        self.assertEqual(agent.state, RefundState.AWAITING_AMOUNT)
        self.assertIn("Which amount", reply.text)
        self.assertTrue(any("[Guard]" in t for t in reply.trace))

    def test_user_claims_do_not_override_account(self):  # GR-03
        agent = make_agent(AccountStore(), 2000)
        reply = agent.handle("My recharge was ₹2000 yesterday, refund ₹2000.")
        self.assertEqual(agent.state, RefundState.REFUND_REJECTED)


if __name__ == "__main__":
    unittest.main()
