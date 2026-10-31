
# Assignment 08 — Project
## Cloud Computing: Supabase and AWS Pricing

## Part 3A: Supabase Setup

### Project Information

- Provider: Supabase
- Project name: python200
- Plan: Free
- Region: East US (North Virginia)
- Database: PostgreSQL
- Project status: Created successfully

### Database Tables

**Table 1: weather_raw**

- date (date, primary key)
- temperature_2m_max (numeric)
- temperature_2m_min (numeric)
- precipitation_sum (numeric)
- wind_speed_10m_max (numeric)
- loaded_at (timestamptz, default now())

**Table 2: weather_enriched**

- date (date, primary key, foreign key to weather_raw.date)
- good_for_running (boolean)
- confidence (numeric)
- llm_summary (text)
- enriched_at (timestamptz, default now())

### Setup Verification

Both tables were created successfully using the
Supabase SQL Editor.

I verified their columns and data types using SQL.
I also verified the primary keys and foreign key.

Row Level Security was disabled as required by the assignment.

### Python Environment

I configured a local .env file with these variables:

- SUPABASE_URL
- SUPABASE_KEY

The .env file is excluded from Git tracking.

I successfully tested loading both variables using
Python 3.13.15 and python-dotenv.

The actual API connection has not yet been tested.

No passwords or API keys are included in this report.


## Part 3B: AWS Pricing Calculator

I used the AWS Pricing Calculator to compare two cloud
computing scenarios in US East (N. Virginia).


### AWS Calculator Estimate Link

https://calculator.aws/#/estimate?id=acf60e3fc69d7231c2a78f3db0cd46367b0a56e9


### Scenario A: Small Development Server

- Service: Amazon EC2
- Instance: t3.micro
- Operating System: Linux
- vCPU: 2
- Memory: 1 GiB
- Pricing: On-Demand
- Usage: 160 hours per month
- Hourly rate: $0.0104

**Estimated monthly cost: $1.66**

### Scenario B: Production Data Pipeline

This scenario uses three AWS services:

**1. GPU Compute (Amazon EC2)**

- Instance: p3.2xlarge
- vCPU: 8
- Memory: 61 GiB
- GPU: 1 NVIDIA V100
- Pricing: On-Demand
- Usage: 730 hours per month
- Hourly rate: $3.06
- Monthly cost: $2,233.80

**2. Database (Amazon RDS for PostgreSQL)**

- Instance: db.m5.large
- vCPU: 2
- Memory: 7.5 GiB
- Deployment: Single-AZ
- Pricing: On-Demand
- Utilization: 100%
- Storage: 100 GB, General Purpose SSD (gp2)
- RDS Proxy: No
- Database Insights: No
- Extended Support: No
- Additional backup storage: 0 GB
- Snapshot export: 0 GB
- Monthly cost: $190.35

**3. Object Storage (Amazon S3)**

- Storage class: S3 Standard
- Storage: 1,024 GB
- Region: US East (N. Virginia)
- Data already stored in S3
- No additional API requests or data transfer included
- Monthly cost: $23.55

### Cost Comparison

| Service | Monthly Cost |
|---------|-------------:|
| Scenario A: EC2 t3.micro | $1.66 |
| Scenario B: EC2 GPU | $2,233.80 |
| Scenario B: RDS PostgreSQL | $190.35 |
| Scenario B: S3 Standard | $23.55 |
| **Scenario B Total** | **$2,447.70** |
| **Combined Monthly Total** | **$2,449.36** |

Combined annual estimate: $29,392.32.

These are estimates, not actual charges.
The estimates exclude additional services and unspecified usage.

### What I Learned

The GPU server is the most expensive component.

Running an expensive GPU instance continuously can make
cloud computing costly. For workloads that do not need
24/7 processing, using the GPU only when needed can
significantly reduce compute costs.

Cloud pricing depends on instance type, usage hours,
storage, region, and additional services.


## Part 3C: Video Walkthrough

I recorded a short video showing my Supabase project,
the two database tables, and my AWS cost estimates.

Video link: https://drive.google.com/file/d/1HUmMqmxpr5O3u0ewPrVQMUGwpaLbuNZE/view?usp=sharing
