output "vm_external_ip" {
  description = "Public IP of the pairlab VM"
  value       = google_compute_instance.pairlab.network_interface[0].access_config[0].nat_ip
}

output "bucket_name" {
  description = "GCS artefact bucket name"
  value       = google_storage_bucket.pairlab.name
}
