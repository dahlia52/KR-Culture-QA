import json
import os

def count_domains(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    domain_counts = {}
    for item in data:
        domain = item['input']['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return domain_counts

def print_domain_distribution(domain_counts, title):
    print(f"\n{title}:")
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    for domain, count in sorted_domains:
        print(f"{domain}: {count} questions")
    print(f"Total: {sum(domain_counts.values())} questions")

# File paths
base_dir = '/data5/wisdomjeong/Korean_Culture_QA_2025/resource/QA/'
files = {
    'Train': 'korean_culture_qa_V1.0_train+.json',
    'Dev': 'korean_culture_qa_V1.0_dev+.json',
    'Test': 'korean_culture_qa_V1.0_test+.json'
}

# Process each file
all_domains = {}
for name, filename in files.items():
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path):
        domain_counts = count_domains(file_path)
        print_domain_distribution(domain_counts, f"{name} Set Domain Distribution")
        
        # Aggregate for combined statistics
        for domain, count in domain_counts.items():
            all_domains[domain] = all_domains.get(domain, 0) + count
    else:
        print(f"\nFile not found: {file_path}")

# Print combined statistics
if all_domains:
    print_domain_distribution(all_domains, "\nCombined Domain Distribution")
