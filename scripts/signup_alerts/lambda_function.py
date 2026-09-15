"""
Cognito Post-confirmation trigger: email an SNS topic when a new user signs up.

Deployed by setup_signup_alerts.py. TOPIC_ARN is set as a Lambda env var.
Cognito requires the handler to return the event unchanged.
"""
import os

import boto3

sns = boto3.client("sns")
TOPIC_ARN = os.environ["TOPIC_ARN"]


def handler(event, context):
    # PostConfirmation also fires on password-reset confirmation; only alert on
    # genuine sign-ups.
    if event.get("triggerSource") != "PostConfirmation_ConfirmSignUp":
        return event

    attrs = event.get("request", {}).get("userAttributes", {})
    email = attrs.get("email", "unknown")
    sub = attrs.get("sub") or event.get("userName", "unknown")

    sns.publish(
        TopicArn=TOPIC_ARN,
        Subject=f"Friday: new user sign-up — {email}",
        Message=(
            "A new user just confirmed their Friday account.\n\n"
            f"Email: {email}\n"
            f"Cognito sub: {sub}\n"
            f"User pool: {event.get('userPoolId', '?')}\n"
        ),
    )
    return event
