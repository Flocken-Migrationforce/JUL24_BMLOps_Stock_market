to read app.log, the logging informations:
in Windows Powershell: Get-Content app.log -Wait
in LInux and macOS: tail -f app.log

2409051804
delete all grafana and prometheus instances, pods, deployments of kubernetes:

kubectl delete -f ./src/kubernetes-deployment.yaml
kubectl delete svc --all
