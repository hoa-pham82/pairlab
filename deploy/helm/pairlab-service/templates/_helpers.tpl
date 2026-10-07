{{- define "svc.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "svc.selector" -}}
app.kubernetes.io/name: {{ .Release.Name }}
{{- end }}
