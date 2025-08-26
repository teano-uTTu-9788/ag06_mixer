#!/usr/bin/env python3
"""
Cloud-Native Kubernetes Manifests Generator
Following Google, Netflix, and Uber best practices for production Kubernetes deployments

Based on:
- Google Kubernetes Engine best practices
- Netflix microservices patterns
- Uber's service mesh architecture
- CNCF recommendations
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import base64

@dataclass
class ResourceRequirements:
    """Resource requirements following GKE best practices"""
    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str
    ephemeral_storage_request: str = "1Gi"
    ephemeral_storage_limit: str = "2Gi"

@dataclass
class SecurityContext:
    """Security context following Google security standards"""
    run_as_non_root: bool = True
    run_as_user: int = 1000
    run_as_group: int = 1000
    fs_group: int = 2000
    allow_privilege_escalation: bool = False
    read_only_root_filesystem: bool = True
    capabilities_drop: List[str] = None
    
    def __post_init__(self):
        if self.capabilities_drop is None:
            self.capabilities_drop = ["ALL"]

class KubernetesManifestGenerator:
    """Generate production-grade Kubernetes manifests"""
    
    def __init__(self, namespace: str = "ag06-production"):
        self.namespace = namespace
        self.labels = {
            "app.kubernetes.io/name": "ag06",
            "app.kubernetes.io/version": "1.0.0",
            "app.kubernetes.io/managed-by": "ag06-deployer",
            "app.kubernetes.io/part-of": "ag06-platform"
        }
    
    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace with proper labels and resource quotas"""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace,
                "labels": {
                    **self.labels,
                    "name": self.namespace,
                    "istio-injection": "enabled"  # Enable Istio service mesh
                },
                "annotations": {
                    "scheduler.alpha.kubernetes.io/node-selector": "workload-type=production"
                }
            }
        }
    
    def generate_resource_quota(self) -> Dict[str, Any]:
        """Generate resource quota following Google best practices"""
        return {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": "ag06-resource-quota",
                "namespace": self.namespace,
                "labels": self.labels
            },
            "spec": {
                "hard": {
                    "requests.cpu": "10",
                    "requests.memory": "20Gi",
                    "requests.storage": "100Gi",
                    "limits.cpu": "20",
                    "limits.memory": "40Gi",
                    "persistentvolumeclaims": "10",
                    "services": "10",
                    "secrets": "20",
                    "configmaps": "20",
                    "pods": "50"
                }
            }
        }
    
    def generate_network_policies(self) -> List[Dict[str, Any]]:
        """Generate network policies for zero-trust networking"""
        
        # Default deny all ingress traffic
        deny_all = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "default-deny-ingress",
                "namespace": self.namespace,
                "labels": self.labels
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress"]
            }
        }
        
        # Allow ingress from ingress controller
        allow_ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy", 
            "metadata": {
                "name": "allow-ingress-controller",
                "namespace": self.namespace,
                "labels": self.labels
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/component": "api"
                    }
                },
                "policyTypes": ["Ingress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "istio-system"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 8080
                            }
                        ]
                    }
                ]
            }
        }
        
        # Allow internal service communication
        allow_internal = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "allow-internal-communication",
                "namespace": self.namespace,
                "labels": self.labels
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": self.namespace
                                    }
                                }
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": self.namespace
                                    }
                                }
                            }
                        ]
                    },
                    # Allow DNS resolution
                    {
                        "to": [],
                        "ports": [
                            {
                                "protocol": "UDP",
                                "port": 53
                            }
                        ]
                    }
                ]
            }
        }
        
        return [deny_all, allow_ingress, allow_internal]
    
    def generate_service_account(self, service_name: str) -> Dict[str, Any]:
        """Generate service account with minimal permissions"""
        return {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": f"ag06-{service_name}",
                "namespace": self.namespace,
                "labels": {
                    **self.labels,
                    "app.kubernetes.io/component": service_name
                },
                "annotations": {
                    "iam.gke.io/gcp-service-account": f"ag06-{service_name}@project-id.iam.gserviceaccount.com"
                }
            },
            "automountServiceAccountToken": False  # Security best practice
        }
    
    def generate_deployment(self, 
                          service_name: str, 
                          image: str,
                          resources: ResourceRequirements,
                          replicas: int = 3,
                          env_vars: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate production-grade deployment"""
        
        container_labels = {
            **self.labels,
            "app.kubernetes.io/component": service_name
        }
        
        security_context = SecurityContext()
        
        container_spec = {
            "name": service_name,
            "image": image,
            "imagePullPolicy": "Always",
            "ports": [
                {
                    "name": "http",
                    "containerPort": 8080,
                    "protocol": "TCP"
                },
                {
                    "name": "metrics", 
                    "containerPort": 9090,
                    "protocol": "TCP"
                }
            ],
            "env": [
                {"name": "SERVICE_NAME", "value": service_name},
                {"name": "ENVIRONMENT", "value": "production"},
                {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                {"name": "POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
                {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}
            ],
            "resources": {
                "requests": {
                    "cpu": resources.cpu_request,
                    "memory": resources.memory_request,
                    "ephemeral-storage": resources.ephemeral_storage_request
                },
                "limits": {
                    "cpu": resources.cpu_limit,
                    "memory": resources.memory_limit,
                    "ephemeral-storage": resources.ephemeral_storage_limit
                }
            },
            "securityContext": {
                "runAsNonRoot": security_context.run_as_non_root,
                "runAsUser": security_context.run_as_user,
                "runAsGroup": security_context.run_as_group,
                "allowPrivilegeEscalation": security_context.allow_privilege_escalation,
                "readOnlyRootFilesystem": security_context.read_only_root_filesystem,
                "capabilities": {
                    "drop": security_context.capabilities_drop
                }
            },
            "livenessProbe": {
                "httpGet": {
                    "path": "/health",
                    "port": "http"
                },
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 3,
                "successThreshold": 1
            },
            "readinessProbe": {
                "httpGet": {
                    "path": "/ready",
                    "port": "http"
                },
                "initialDelaySeconds": 5,
                "periodSeconds": 5,
                "timeoutSeconds": 3,
                "failureThreshold": 3,
                "successThreshold": 1
            },
            "startupProbe": {
                "httpGet": {
                    "path": "/startup",
                    "port": "http"
                },
                "initialDelaySeconds": 10,
                "periodSeconds": 10,
                "timeoutSeconds": 3,
                "failureThreshold": 30,
                "successThreshold": 1
            },
            "volumeMounts": [
                {
                    "name": "tmp",
                    "mountPath": "/tmp"
                },
                {
                    "name": "var-cache",
                    "mountPath": "/var/cache"
                },
                {
                    "name": "config",
                    "mountPath": "/etc/ag06",
                    "readOnly": True
                }
            ]
        }
        
        # Add custom environment variables
        if env_vars:
            for key, value in env_vars.items():
                container_spec["env"].append({"name": key, "value": value})
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"ag06-{service_name}",
                "namespace": self.namespace,
                "labels": container_labels,
                "annotations": {
                    "deployment.kubernetes.io/revision": "1"
                }
            },
            "spec": {
                "replicas": replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "ag06",
                        "app.kubernetes.io/component": service_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": container_labels,
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "9090",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": f"ag06-{service_name}",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": security_context.run_as_user,
                            "runAsGroup": security_context.run_as_group,
                            "fsGroup": security_context.fs_group
                        },
                        "containers": [container_spec],
                        "volumes": [
                            {
                                "name": "tmp",
                                "emptyDir": {}
                            },
                            {
                                "name": "var-cache",
                                "emptyDir": {}
                            },
                            {
                                "name": "config",
                                "configMap": {
                                    "name": f"ag06-{service_name}-config"
                                }
                            }
                        ],
                        "affinity": {
                            "podAntiAffinity": {
                                "preferredDuringSchedulingIgnoredDuringExecution": [
                                    {
                                        "weight": 100,
                                        "podAffinityTerm": {
                                            "labelSelector": {
                                                "matchLabels": {
                                                    "app.kubernetes.io/name": "ag06",
                                                    "app.kubernetes.io/component": service_name
                                                }
                                            },
                                            "topologyKey": "kubernetes.io/hostname"
                                        }
                                    }
                                ]
                            }
                        },
                        "terminationGracePeriodSeconds": 30,
                        "dnsPolicy": "ClusterFirst",
                        "restartPolicy": "Always"
                    }
                }
            }
        }
    
    def generate_service(self, service_name: str, port: int = 8080) -> Dict[str, Any]:
        """Generate Kubernetes service"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"ag06-{service_name}",
                "namespace": self.namespace,
                "labels": {
                    **self.labels,
                    "app.kubernetes.io/component": service_name
                },
                "annotations": {
                    "service.beta.kubernetes.io/load-balancer-source-ranges": "10.0.0.0/8",
                    "prometheus.io/scrape": "true",
                    "prometheus.io/port": "9090"
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "name": "http",
                        "port": port,
                        "targetPort": "http",
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9090,
                        "targetPort": "metrics",
                        "protocol": "TCP"
                    }
                ],
                "selector": {
                    "app.kubernetes.io/name": "ag06",
                    "app.kubernetes.io/component": service_name
                }
            }
        }
    
    def generate_horizontal_pod_autoscaler(self, service_name: str, 
                                         min_replicas: int = 3,
                                         max_replicas: int = 100) -> Dict[str, Any]:
        """Generate HPA following Google best practices"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"ag06-{service_name}-hpa",
                "namespace": self.namespace,
                "labels": {
                    **self.labels,
                    "app.kubernetes.io/component": service_name
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"ag06-{service_name}"
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource", 
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 15
                            },
                            {
                                "type": "Pods",
                                "value": 4,
                                "periodSeconds": 15
                            }
                        ],
                        "selectPolicy": "Max"
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 10,
                                "periodSeconds": 60
                            }
                        ],
                        "selectPolicy": "Min"
                    }
                }
            }
        }
    
    def generate_pod_disruption_budget(self, service_name: str) -> Dict[str, Any]:
        """Generate PodDisruptionBudget for high availability"""
        return {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget", 
            "metadata": {
                "name": f"ag06-{service_name}-pdb",
                "namespace": self.namespace,
                "labels": {
                    **self.labels,
                    "app.kubernetes.io/component": service_name
                }
            },
            "spec": {
                "minAvailable": "50%",  # Keep at least 50% available during disruptions
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "ag06",
                        "app.kubernetes.io/component": service_name
                    }
                }
            }
        }
    
    def generate_ingress(self, hostname: str = "ag06.example.com") -> Dict[str, Any]:
        """Generate ingress with TLS termination"""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "ag06-ingress",
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": {
                    "kubernetes.io/ingress.class": "gce",
                    "kubernetes.io/ingress.global-static-ip-name": "ag06-ip",
                    "ingress.gcp.kubernetes.io/managed-certificates": "ag06-ssl-cert",
                    "kubernetes.io/ingress.allow-http": "false"
                }
            },
            "spec": {
                "tls": [
                    {
                        "secretName": "ag06-tls-secret",
                        "hosts": [hostname]
                    }
                ],
                "rules": [
                    {
                        "host": hostname,
                        "http": {
                            "paths": [
                                {
                                    "path": "/api/*",
                                    "pathType": "ImplementationSpecific",
                                    "backend": {
                                        "service": {
                                            "name": "ag06-api",
                                            "port": {
                                                "number": 8080
                                            }
                                        }
                                    }
                                },
                                {
                                    "path": "/dashboard/*",
                                    "pathType": "ImplementationSpecific", 
                                    "backend": {
                                        "service": {
                                            "name": "ag06-dashboard",
                                            "port": {
                                                "number": 8080
                                            }
                                        }
                                    }
                                },
                                {
                                    "path": "/*",
                                    "pathType": "ImplementationSpecific",
                                    "backend": {
                                        "service": {
                                            "name": "ag06-frontend",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

def generate_complete_kubernetes_deployment():
    """Generate complete production Kubernetes deployment"""
    generator = KubernetesManifestGenerator()
    
    # Service configurations
    services = {
        "api": {
            "image": "gcr.io/ag06-project/api:latest",
            "resources": ResourceRequirements(
                cpu_request="500m",
                memory_request="1Gi", 
                cpu_limit="2000m",
                memory_limit="4Gi"
            ),
            "replicas": 5,
            "env_vars": {
                "LOG_LEVEL": "INFO",
                "MAX_WORKERS": "4"
            }
        },
        "dashboard": {
            "image": "gcr.io/ag06-project/dashboard:latest",
            "resources": ResourceRequirements(
                cpu_request="200m",
                memory_request="512Mi",
                cpu_limit="1000m", 
                memory_limit="2Gi"
            ),
            "replicas": 3,
            "env_vars": {
                "NODE_ENV": "production"
            }
        },
        "workflow-engine": {
            "image": "gcr.io/ag06-project/workflow-engine:latest",
            "resources": ResourceRequirements(
                cpu_request="1000m",
                memory_request="2Gi",
                cpu_limit="4000m",
                memory_limit="8Gi"
            ),
            "replicas": 3,
            "env_vars": {
                "WORKER_THREADS": "8",
                "MAX_QUEUE_SIZE": "10000"
            }
        },
        "ml-analytics": {
            "image": "gcr.io/ag06-project/ml-analytics:latest", 
            "resources": ResourceRequirements(
                cpu_request="2000m",
                memory_request="4Gi",
                cpu_limit="8000m",
                memory_limit="16Gi"
            ),
            "replicas": 2,
            "env_vars": {
                "MODEL_CACHE_SIZE": "1000",
                "BATCH_SIZE": "32"
            }
        }
    }
    
    manifests = []
    
    # Generate core infrastructure
    manifests.append(generator.generate_namespace())
    manifests.append(generator.generate_resource_quota())
    manifests.extend(generator.generate_network_policies())
    
    # Generate per-service resources
    for service_name, config in services.items():
        manifests.append(generator.generate_service_account(service_name))
        manifests.append(generator.generate_deployment(
            service_name=service_name,
            image=config["image"],
            resources=config["resources"],
            replicas=config["replicas"],
            env_vars=config["env_vars"]
        ))
        manifests.append(generator.generate_service(service_name))
        manifests.append(generator.generate_horizontal_pod_autoscaler(service_name))
        manifests.append(generator.generate_pod_disruption_budget(service_name))
    
    # Generate ingress
    manifests.append(generator.generate_ingress())
    
    return manifests

def export_manifests_to_files():
    """Export all manifests to separate YAML files"""
    manifests = generate_complete_kubernetes_deployment()
    
    output_dir = Path("k8s-manifests")
    output_dir.mkdir(exist_ok=True)
    
    for i, manifest in enumerate(manifests):
        kind = manifest.get("kind", "Unknown")
        name = manifest.get("metadata", {}).get("name", f"resource-{i}")
        
        filename = f"{i:02d}-{kind.lower()}-{name}.yaml"
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Generated: {filepath}")
    
    # Generate kustomization.yaml
    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "namespace": "ag06-production",
        "resources": [f.name for f in output_dir.glob("*.yaml") if f.name != "kustomization.yaml"],
        "commonLabels": {
            "app.kubernetes.io/managed-by": "kustomize"
        },
        "images": [
            {
                "name": "gcr.io/ag06-project/api",
                "newTag": "v1.0.0"
            },
            {
                "name": "gcr.io/ag06-project/dashboard", 
                "newTag": "v1.0.0"
            },
            {
                "name": "gcr.io/ag06-project/workflow-engine",
                "newTag": "v1.0.0"
            },
            {
                "name": "gcr.io/ag06-project/ml-analytics",
                "newTag": "v1.0.0"
            }
        ]
    }
    
    with open(output_dir / "kustomization.yaml", "w") as f:
        yaml.dump(kustomization, f, default_flow_style=False)
    
    print(f"✅ Generated: {output_dir / 'kustomization.yaml'}")
    
    return output_dir

if __name__ == "__main__":
    print("🚀 Generating Cloud-Native Kubernetes Manifests")
    print("=" * 60)
    
    output_directory = export_manifests_to_files()
    
    print(f"\n📁 All manifests generated in: {output_directory}")
    print(f"📄 Total files: {len(list(output_directory.glob('*.yaml')))}")
    print(f"\n🔧 To deploy:")
    print(f"   kubectl apply -k {output_directory}")
    print(f"\n✅ Cloud-Native Kubernetes deployment ready!")