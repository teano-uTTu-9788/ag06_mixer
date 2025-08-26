# Route 53 Hosted Zone
resource "aws_route53_zone" "main" {
  name = "ag06mixer.com"

  tags = {
    Name = "ag06mixer-hosted-zone"
  }
}

# SSL Certificate
resource "aws_acm_certificate" "main" {
  domain_name               = "ag06mixer.com"
  subject_alternative_names = [
    "*.ag06mixer.com",
    "api.ag06mixer.com",
    "monitor.ag06mixer.com"
  ]
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "ag06mixer-ssl-cert"
  }
}

# Certificate validation
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main.zone_id
}

resource "aws_acm_certificate_validation" "main" {
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}
