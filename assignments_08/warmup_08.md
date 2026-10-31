
# Assignment 08 — Warmup
## Introduction to Cloud Computing

## Part 1: Cloud Concepts

### Question 1: Cloud Economics

With cloud computing, if you need more computing power, you can get it quickly and pay for what you use. Unlike owning your own servers, you don't need to buy and maintain expensive hardware.

### Question 2: Vertical and Horizontal Scaling

Vertical scaling means adding more computing resources (CPU, RAM, or storage) to a single, existing server. It makes your current machine more powerful.

Horizontal scaling means adding more servers to share the workload.

**Scenario 1 — Horizontal Scaling:** We can add more servers to handle the increased number of users.

**Scenario 2 — Vertical Scaling:** We can upgrade the existing server with a faster GPU and more RAM to train the model faster.

**Scenario 3 — Horizontal Scaling:** We can add more servers to process the files in parallel and finish the work faster.

### Question 3: Cloud Service Models

**Classifications:**

1. Gmail — SaaS: It is a ready-to-use email service.
2. Azure Virtual Machines — IaaS: It provides virtual servers that users can manage.
3. AWS S3 — IaaS: It provides cloud storage without buying physical storage hardware.
4. GitHub Codespaces — PaaS: It provides a ready-to-use cloud development environment.
5. Snowflake — SaaS: It provides a managed platform for storing and analyzing data.
6. Supabase — BaaS: It provides backend services such as databases and authentication.

**IaaS (Infrastructure as a Service):**

It is like renting a blank computer or virtual server over the internet. The provider gives you virtual hardware, storage, and networking. For example, Azure Virtual Machines. You are responsible for managing the operating system, software, and applications.

**PaaS (Platform as a Service):**

It provides a ready-to-use environment. The cloud provider manages the servers, operating system, and runtime environment for you. For example, Render. You are responsible for your application code, data, and settings.

**SaaS (Software as a Service):**

It is a fully finished, ready-to-use software application delivered over the internet. You do not need to build or maintain the application yourself. For example, Gmail. You are responsible for managing your account, settings, and data.

### Question 4: Managed Data Platforms

A managed data platform is a ready-to-use system for storing and analyzing data. For example, Snowflake and Databricks provide many tools without needing to manage the servers yourself.

Compared to using AWS or GCP services directly, it is usually easier and faster to start. However, you have less control over the infrastructure and may have to pay more.

### Question 5: When Cloud Is Not the Best Choice

Cloud computing is not the right choice for small, simple tasks that already run perfectly on your personal computer.

Moving these basic programs to the cloud can add unnecessary cost and complexity without providing significant benefits.

## Part 2: Cloud Provider Landscape

### Question 1: The Three Hyperscalers

**AWS (Amazon Web Services):**

AWS provides a wide variety of cloud tools, making it a powerful choice for both growing startups and large global enterprises.

**Microsoft Azure:**

Microsoft Azure integrates well with Microsoft products, making it a good choice for established businesses that already use Microsoft software.

**Google Cloud Platform (GCP):**

Google Cloud offers powerful artificial intelligence and data analysis tools, making it a good choice for technology companies focused on big data and machine learning.

### Question 2: Why Supabase Instead of Azure?

**Reason 1 — Simpler Setup:**

Supabase provides a ready-to-use database and API in just a few clicks. In contrast, Azure is a large enterprise cloud platform that may require more configuration of networking, security, and resources, depending on the services you choose.

**Reason 2 — Accessible Free Tier:**

Supabase is highly accessible because it offers a free tier, which is useful for learning and small projects. Azure also offers free services, but its wide range of pricing options can make cost management more complicated for beginners.

**Reason 3 — PostgreSQL and Familiar Tools:**

Supabase is built on PostgreSQL, a widely used open-source database. This makes it easy to work with familiar SQL and Python tools to extract, transform, and load (ETL) data without learning additional cloud services.

**My Reflection:**

This suggests that when starting a new project, you should not automatically pick the biggest or most famous enterprise cloud.

Instead, you should choose a tool that matches the actual size, budget, and complexity of your task.

A simpler tool that lets you build quickly is often a better choice than a massive platform filled with advanced features you may never use.

### Question 3: Choosing Cloud Services

**Scenario 1 — Storing 10 TB of Images:**

I would use Object Storage, such as Amazon S3, to store 10 TB of images and access them from different computers.

**Scenario 2 — Training an ML Model for Four Hours:**

I would use a GPU instance, such as AWS EC2 with a GPU, to train my machine learning model for four hours.

**Scenario 3 — Automatically Scaling a Web API:**

I would use a serverless service, such as Google Cloud Run, to automatically scale my web API based on traffic.

**Scenario 4 — Generating Text with an LLM:**

I would use an LLM API, such as the OpenAI API, to send structured data to a language model and receive a text response.

### Question 4: My Cloud Data Project

**My Project:**

I want to build a Python web application that analyzes California real estate data to calculate property taxes and investment returns.

It will allow users to upload property information and receive an automated financial report.

**Cloud Services:**

I would use Google Cloud Run to run the Python code and process the real estate calculations only when a user requests a report, saving money when the app is not in use.

I would also use Supabase PostgreSQL to store property information and calculation results.

**Trade-offs:**

Using a single cloud provider makes it easier to connect your database, storage, and compute services with built-in security features.

However, it may limit your choices and make you more dependent on one provider.
