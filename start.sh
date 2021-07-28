#!/bin/bash
helm uninstall automl-cluster
helm install automl-cluster ./ray
docker-compose build
docker-compose push
kubectl config set-context --current --namespace=fraxses-fraxses-dev
kubectl apply -f deployments/tune
kubectl apply -f deployments/serve
kubectl rollout restart deployment automl-tune
kubectl rollout restart deployment automl-serve
kubectl get pods | grep automl