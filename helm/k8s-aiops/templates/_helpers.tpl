{{- define "k8s-aiops.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "k8s-aiops.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "k8s-aiops.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "k8s-aiops.labels" -}}
app.kubernetes.io/name: {{ include "k8s-aiops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "k8s-aiops.selectorLabels" -}}
app.kubernetes.io/name: {{ include "k8s-aiops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "k8s-aiops.serviceAccountName" -}}
{{ include "k8s-aiops.fullname" . }}
{{- end -}}

{{- define "k8s-aiops.smtpSecretName" -}}
{{- if .Values.smtp.existingSecret -}}
{{ .Values.smtp.existingSecret }}
{{- else -}}
{{ include "k8s-aiops.fullname" . }}-smtp
{{- end -}}
{{- end -}}
