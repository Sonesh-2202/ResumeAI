"""
NEW: Text analysis utilities — keyword extraction, gap analysis, and anonymization.
"""

import re


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords/phrases from text using noun and skill patterns.
    
    Args:
        text: Input text (resume or job description).
        
    Returns:
        Set of extracted keywords.
    """
    if not text:
        return set()
    
    keywords: set[str] = set()
    lower_text = text.lower()
    
    # Technical skills and tools
    tech_patterns = re.findall(
        r'\b(?:Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Scala|R|Objective-C|'
        r'React|Angular|Vue|Next\.?js|Svelte|Ember|Backbone|'
        r'Node\.?js|Django|Flask|FastAPI|Spring|Express|Rails|Laravel|Asp\.?Net|'
        r'AWS|Azure|GCP|Google Cloud|Oracle Cloud|'
        r'Docker|Kubernetes|K8s|Terraform|Ansible|Puppet|Chef|CloudFormation|'
        r'Jenkins|GitHub|GitLab|Bitbucket|Azure DevOps|Travis|CircleCI|'
        r'PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|Firestore|'
        r'RabbitMQ|Kafka|ActiveMQ|SQS|Pub/Sub|'
        r'TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Spark|Hadoop|Flink|'
        r'Linux|Unix|Windows|macOS|Android|iOS|'
        r'Nginx|Apache|Tomcat|IIS|'
        r'GraphQL|REST|SOAP|gRPC|WebSocket|'
        r'OAuth|JWT|SAML|LDAP|Active Directory|'
        r'SSL|TLS|HTTPS|HTTP/2|'
        r'HTML5?|CSS|SASS|Less|Webpack|Babel|Gulp|Grunt|'
        r'Git|SVN|Mercurial|Perforce|'
        r'Jira|Confluence|Trello|Asana|Monday|'
        r'Tableau|Power\s*BI|Looker|Metabase|Grafana|'
        r'Figma|Sketch|Adobe|Photoshop|Illustrator|XD|'
        r'Slack|Teams|Discord|Zoom|Asana|'
        r'Machine Learning|Deep Learning|NLP|Computer Vision|AI|'
        r'DevOps|SRE|Site Reliability Engineering|'
        r'CI/CD|Continuous Integration|Continuous Deployment|'
        r'Agile|Scrum|Kanban|SAFe|Lean|'
        r'API Design|Microservices|Monolith|SOA|'
        r'Database Design|SQL|NoSQL|RDBMS|'
        r'Testing|Unit Test|Integration Test|E2E|Functional Test|'
        r'Security|Encryption|Authentication|Authorization|'
        r'Monitoring|Logging|Observability|Tracing|'
        r'Performance Optimization|Caching|Load Balancing|'
        r'Data Pipeline|ETL|ELT|Data Warehouse|Data Lake|'
        r'Analytics|Business Intelligence|Reporting|Dashboards|'
        r'Project Management|Product Management|Technical Leadership|'
        r'Mentoring|Training|Knowledge Transfer|Documentation|'
        r'SOAP|REST API|Webhooks|Message Queue|Event Driven|Pub-Sub|'
        r'Blockchain|Cryptocurrency|Smart Contracts|Web3|DeFi|NFT|'
        r'Mobile App|Web App|Desktop App|CLI|SPA|PWA|'
        r'Accessibility|WCAG|A11y|Internationalization|i18n|Localization|l10n)\b',
        text,
        re.IGNORECASE,
    )
    for term in tech_patterns:
        keywords.add(term.lower().strip())
    
    # Soft skills and domain keywords
    soft_keywords = re.findall(
        r'\b(?:communication|leadership|collaboration|teamwork|problem[- ]solving|'
        r'critical thinking|creativity|innovation|adaptability|time management|'
        r'organization|attention to detail|reliability|accountability|initiative|'
        r'strategic thinking|planning|analysis|presentation|negotiation|'
        r'mentoring|coaching|training|customer service|user experience|ux|ui|'
        r'business acumen|financial|budget|roi|kpi|okt|agile|scrum|kanban|'
        r'sales|marketing|branding|seo|sem|content|copywriting|'
        r'hr|recruiting|talent|onboarding|performance management)\b',
        text,
        re.IGNORECASE,
    )
    for kw in soft_keywords:
        keywords.add(kw.lower().strip())
    
    # Degree and certification keywords
    cert_keywords = re.findall(
        r'\b(?:bachelor|master|phd|doctorate|associate|diploma|certification|'
        r'aws certified|google cloud certified|azure certified|cisco|comptia|'
        r'pmp|cissp|ccna|ccie|certified kubernetes|ckad|cka|scrum master|csm|'
        r'salesforce|servicenow|sap|oracle|databricks|snowflake)\b',
        text,
        re.IGNORECASE,
    )
    for cert in cert_keywords:
        keywords.add(cert.lower().strip())
    
    return keywords


def find_keyword_gaps(
    job_description: str,
    resume_text: str,
) -> tuple[set[str], set[str], float]:
    """
    Find keywords present in JD but missing from resume.
    
    Args:
        job_description: The job description text.
        resume_text: The resume text.
        
    Returns:
        Tuple of (jd_keywords, resume_keywords, coverage_percentage).
        coverage_percentage = (len(intersection) / len(jd_keywords)) * 100
    """
    jd_kw = extract_keywords(job_description)
    res_kw = extract_keywords(resume_text)
    
    # Calculate coverage
    if not jd_kw:
        coverage = 100.0
    else:
        intersection = jd_kw & res_kw
        coverage = (len(intersection) / len(jd_kw)) * 100.0
    
    return jd_kw, res_kw, coverage


def anonymize_resume(resume_text: str) -> str:
    """
    Remove personally identifying information from resume text.
    
    Removes: names, emails, phone numbers, LinkedIn URLs, GitHub URLs.
    
    Args:
        resume_text: Original resume text.
        
    Returns:
        Anonymized resume text.
    """
    if not resume_text:
        return resume_text
    
    result = resume_text
    
    # Remove email addresses
    result = re.sub(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        '[EMAIL]',
        result,
    )
    
    # Remove phone numbers (US format and international)
    result = re.sub(
        r'(?:\+?1[-.]?)?\(?(?:\d{3})\)?[-.\s]?(?:\d{3})[-.\s]?(?:\d{4})\b',
        '[PHONE]',
        result,
    )
    result = re.sub(
        r'\+\d{1,3}(?:\s?\d{1,14})',
        '[PHONE]',
        result,
    )
    
    # Remove LinkedIn URLs
    result = re.sub(
        r'https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-]+/?',
        '[LINKEDIN]',
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r'linkedin\.com/in/[a-zA-Z0-9\-]+',
        '[LINKEDIN]',
        result,
        flags=re.IGNORECASE,
    )
    
    # Remove GitHub URLs
    result = re.sub(
        r'https?://(?:www\.)?github\.com/[a-zA-Z0-9\-]+/?',
        '[GITHUB]',
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r'github\.com/[a-zA-Z0-9\-]+',
        '[GITHUB]',
        result,
        flags=re.IGNORECASE,
    )
    
    # Remove Twitter/X handles
    result = re.sub(
        r'@[a-zA-Z0-9_]{1,15}\b',
        '[TWITTER]',
        result,
    )
    
    # Remove postal addresses (generic: lines with common address patterns)
    result = re.sub(
        r'\b\d{1,5}\s+[a-zA-Z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|court|ct|circle|cir|plaza|square|trail|way|parkway|pkwy|circle|cir|hill|hills|mount|mt|pike|terrace|tr)\.?,?\s+[a-zA-Z\s]+,?\s+[a-zA-Z]{2}\s+\d{5}(?:-\d{4})?\b',
        '[ADDRESS]',
        result,
        flags=re.IGNORECASE,
    )
    
    return result


def extract_candidate_name(resume_text: str) -> str:
    """
    Attempt to extract candidate name from resume text (typically first line or header).
    
    Args:
        resume_text: Resume text.
        
    Returns:
        Guessed name or empty string.
    """
    if not resume_text:
        return ""
    
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    if not lines:
        return ""
    
    # First line is often the name
    first = lines[0]
    # Remove common junk patterns
    first = re.sub(r'(?:resume|cv|curriculum|vitae).*', '', first, flags=re.IGNORECASE).strip()
    
    if len(first) > 2 and len(first) < 80:
        return first
    
    return ""
