terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── GCS bucket for pairlab artefacts ──────────────────────────────────────────

resource "google_storage_bucket" "pairlab" {
  name          = "${var.project_id}-pairlab-artefacts"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition { age = 7 }
    action    { type = "Delete" }
  }
}

# ── VPC + subnet ──────────────────────────────────────────────────────────────

resource "google_compute_network" "pairlab" {
  name                    = "pairlab-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "pairlab" {
  name          = "pairlab-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.pairlab.id
}

# ── Firewall: SSH + app ports ──────────────────────────────────────────────────

resource "google_compute_firewall" "allow_ssh" {
  name    = "pairlab-allow-ssh"
  network = google_compute_network.pairlab.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["pairlab"]
}

resource "google_compute_firewall" "allow_app" {
  name    = "pairlab-allow-app"
  network = google_compute_network.pairlab.name

  allow {
    protocol = "tcp"
    # NGINX gateway (8443), Grafana (3000), Airflow (8083)
    ports = ["8443", "3000", "8083"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["pairlab"]
}

# ── VM ────────────────────────────────────────────────────────────────────────

resource "google_compute_instance" "pairlab" {
  name         = "pairlab-node"
  machine_type = var.machine_type
  zone         = "${var.region}-a"
  tags         = ["pairlab"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 50
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.pairlab.id
    access_config {}  # ephemeral public IP
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    apt-get update -q
    apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release
    # Docker
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
      > /etc/apt/sources.list.d/docker.list
    apt-get update -q
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    usermod -aG docker debian
  EOT

  service_account {
    scopes = ["cloud-platform"]
  }
}
