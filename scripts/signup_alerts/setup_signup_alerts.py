"""
One-time setup: email alerts on new Friday user sign-ups.

Creates (idempotently — safe to re-run):
  1. SNS topic `friday-signup-alerts` with an email subscription.
  2. IAM role for the Lambda (CloudWatch logs + publish to that one topic).
  3. Lambda `friday-signup-alert` from lambda_function.py (same directory).
  4. Permission for Cognito to invoke the Lambda.
  5. Post-confirmation trigger on the user pool, preserving all existing
     pool settings (UpdateUserPool resets any field you omit, so we copy the
     current config and only merge in LambdaConfig.PostConfirmation).

Usage:
  python setup_signup_alerts.py --email you@example.com
  python setup_signup_alerts.py --email you@example.com --dry-run

Requires AWS credentials (~/.aws/credentials or env vars) with permissions on
SNS, IAM, Lambda, and Cognito IDP.
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

DEFAULT_POOL_ID = "us-west-1_b9VdZKUiu"
DEFAULT_REGION = "us-west-1"
TOPIC_NAME = "friday-signup-alerts"
FUNCTION_NAME = "friday-signup-alert"
ROLE_NAME = "friday-signup-alert-lambda-role"

# Mutable UpdateUserPool fields that must be copied from DescribeUserPool,
# otherwise the update silently resets them to defaults.
POOL_PRESERVE_FIELDS = [
    "Policies", "DeletionProtection", "AutoVerifiedAttributes",
    "SmsVerificationMessage", "EmailVerificationMessage",
    "EmailVerificationSubject", "VerificationMessageTemplate",
    "SmsAuthenticationMessage", "UserAttributeUpdateSettings",
    "MfaConfiguration", "DeviceConfiguration", "EmailConfiguration",
    "SmsConfiguration", "UserPoolTags", "AdminCreateUserConfig",
    "UserPoolAddOns", "AccountRecoverySetting",
]


def ensure_topic(sns, email: str) -> str:
    topic_arn = sns.create_topic(Name=TOPIC_NAME)["TopicArn"]  # idempotent
    print(f"[1/5] SNS topic: {topic_arn}")

    subs = sns.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]
    if any(s["Endpoint"] == email and s["Protocol"] == "email" for s in subs):
        print(f"      {email} already subscribed")
    else:
        sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)
        print(f"      Subscribed {email} — CHECK YOUR INBOX and click "
              f"'Confirm subscription' (from no-reply@sns.amazonaws.com)")
    return topic_arn


def ensure_role(iam, topic_arn: str) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        role_arn = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Friday sign-up alert Lambda: logs + SNS publish",
        )["Role"]["Arn"]
        created = True
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        created = False

    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="publish-signup-alerts",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": "sns:Publish",
                "Resource": topic_arn,
            }],
        }),
    )
    print(f"[2/5] IAM role: {role_arn} ({'created' if created else 'exists'})")
    return role_arn


def build_zip() -> bytes:
    src = Path(__file__).parent / "lambda_function.py"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lambda_function.py", src.read_text())
    return buf.getvalue()


def ensure_lambda(lam, role_arn: str, topic_arn: str) -> str:
    code = build_zip()
    kwargs = dict(
        Runtime="python3.12",
        Handler="lambda_function.handler",
        Timeout=10,
        Environment={"Variables": {"TOPIC_ARN": topic_arn}},
    )
    # New IAM roles take a few seconds to become assumable by Lambda.
    for attempt in range(8):
        try:
            fn_arn = lam.create_function(
                FunctionName=FUNCTION_NAME, Role=role_arn,
                Code={"ZipFile": code}, **kwargs,
            )["FunctionArn"]
            print(f"[3/5] Lambda created: {fn_arn}")
            return fn_arn
        except ClientError as e:
            err = e.response["Error"]["Code"]
            if err == "ResourceConflictException":  # exists → update in place
                lam.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=code)
                waiter = lam.get_waiter("function_updated")
                waiter.wait(FunctionName=FUNCTION_NAME)
                lam.update_function_configuration(
                    FunctionName=FUNCTION_NAME, Role=role_arn, **kwargs)
                fn_arn = lam.get_function(FunctionName=FUNCTION_NAME)[
                    "Configuration"]["FunctionArn"]
                print(f"[3/5] Lambda updated: {fn_arn}")
                return fn_arn
            if err == "InvalidParameterValueException" and attempt < 7:
                print("      waiting for IAM role to propagate...")
                time.sleep(5)
                continue
            raise
    raise RuntimeError("Lambda creation failed after retries")


def allow_cognito_invoke(lam, fn_arn: str, pool_arn: str) -> None:
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="cognito-post-confirmation",
            Action="lambda:InvokeFunction",
            Principal="cognito-idp.amazonaws.com",
            SourceArn=pool_arn,
        )
        print("[4/5] Cognito granted permission to invoke the Lambda")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":
            raise
        print("[4/5] Cognito invoke permission already in place")


def attach_trigger(cognito, pool_id: str, fn_arn: str) -> None:
    pool = cognito.describe_user_pool(UserPoolId=pool_id)["UserPool"]
    update = {k: pool[k] for k in POOL_PRESERVE_FIELDS if k in pool}
    lambda_config = dict(pool.get("LambdaConfig") or {})
    if lambda_config.get("PostConfirmation") == fn_arn:
        print("[5/5] Post-confirmation trigger already attached")
        return
    lambda_config["PostConfirmation"] = fn_arn
    cognito.update_user_pool(
        UserPoolId=pool_id, LambdaConfig=lambda_config, **update)
    print("[5/5] Post-confirmation trigger attached to user pool")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--email", required=True, help="alert recipient")
    ap.add_argument("--pool-id", default=DEFAULT_POOL_ID)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify credentials/pool access and exit")
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    account = ident["Account"]
    print(f"AWS account {account}, caller {ident['Arn']}")

    cognito = session.client("cognito-idp")
    pool = cognito.describe_user_pool(UserPoolId=args.pool_id)["UserPool"]
    print(f"User pool: {pool['Name']} ({args.pool_id}), "
          f"{pool.get('EstimatedNumberOfUsers', '?')} users")
    if args.dry_run:
        print("Dry run: credentials and pool access OK.")
        return 0

    pool_arn = pool["Arn"]
    topic_arn = ensure_topic(session.client("sns"), args.email)
    role_arn = ensure_role(session.client("iam"), topic_arn)
    fn_arn = ensure_lambda(session.client("lambda"), role_arn, topic_arn)
    allow_cognito_invoke(session.client("lambda"), fn_arn, pool_arn)
    attach_trigger(cognito, args.pool_id, fn_arn)

    print("\nDone. Confirm the SNS subscription email if you haven't yet — "
          "alerts are not delivered until you do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
