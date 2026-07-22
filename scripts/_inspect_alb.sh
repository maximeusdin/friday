#!/usr/bin/env bash
# Temporary diagnostic: dump ALB listeners + rules for the friday API.
set -euo pipefail
REGION=us-west-1

aws elbv2 describe-load-balancers --region $REGION \
  --query 'LoadBalancers[].{Name:LoadBalancerName,DNS:DNSName,Arn:LoadBalancerArn}' --output json

LB_ARN=$(aws elbv2 describe-load-balancers --region $REGION \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

for L in $(aws elbv2 describe-listeners --region $REGION --load-balancer-arn "$LB_ARN" \
    --query 'Listeners[].ListenerArn' --output text); do
  echo "=== listener: $L"
  aws elbv2 describe-listeners --region $REGION --listener-arns "$L" \
    --query 'Listeners[0].{Port:Port,Proto:Protocol,Default:DefaultActions}' --output json
  aws elbv2 describe-rules --region $REGION --listener-arn "$L" \
    --query 'Rules[].{Prio:Priority,Cond:Conditions,Actions:Actions[].{Type:Type,Fixed:FixedResponseConfig,TG:TargetGroupArn}}' \
    --output json
done
