# Acme Platform Engineering Handbook

This handbook is a fictional sample document used to test and evaluate DocuMind.

## Deployments

Every service is deployed through the shared CI/CD pipeline. A deployment starts when a pull request is merged into the main branch. The pipeline runs unit tests, builds a container image, scans it for vulnerabilities, and pushes it to the registry at registry.acme.example.

New versions are first released as a canary to 5% of production traffic. If the error rate stays below 1% for 20 minutes, the release is promoted to 100% of traffic automatically. Teams are expected to deploy at least once per working day.

## Rollbacks

A rollback must be started within 15 minutes of a failed deployment. The command `acmectl rollback <service>` restores the previous image tag. After a rollback, the on-call engineer opens an incident ticket and writes a short postmortem within 3 working days.

## Containers

All container images must be smaller than 300 MB. Images must run as a non-root user and must expose a health endpoint at /healthz. Base images are limited to the approved slim variants maintained by the platform team. Image tags use the Git commit SHA and never use the latest tag.

## Kubernetes

Workloads run on Kubernetes in two clusters: staging-eu for pre-production testing and prod-eu for production. Every deployment must define CPU and memory requests and limits. The default horizontal autoscaler targets 70% CPU utilisation, with a minimum of 2 replicas and a maximum of 10 replicas.

## Infrastructure as Code

Cloud resources are provisioned only with Terraform. State is stored in a remote bucket with locking enabled, and engineers must never edit resources manually in the cloud console. Server configuration, such as hardening the bastion hosts, is handled with Ansible playbooks stored in the infra-config repository.

## Monitoring and Logging

Metrics are collected by Prometheus and shown in Grafana dashboards. An alert pages the on-call engineer when the error rate rises above 2% for 5 minutes. Application logs are shipped to the central logging stack and retained for 30 days. Logs must be written as JSON and must never contain passwords or personal data.

## On-call

The on-call rotation changes every week on Monday at 10:00 UTC. The primary on-call engineer must acknowledge a page within 10 minutes; otherwise the page escalates to the secondary engineer.

## Troubleshooting a crashing pod

First run `kubectl describe pod` to check events such as OOMKilled or failed probes. Then read the previous container's output with `kubectl logs --previous`. If the cause is a bad release, roll back first and investigate afterwards. Compare the pod's memory usage in Grafana with its configured limit before raising the limit.
