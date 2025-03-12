def extract_company_id(job_detail):
    """
    Extracts the company_id from the job_detail dictionary.

    Args:
        job_detail (dict): The job detail dictionary containing company information.

    Returns:
        str or None: The extracted company_id, or None if not found.
    """
    company_details = job_detail.get("companyDetails", {})
    
    for company_info in company_details.values():
        # Check if 'companyResolutionResult' exists as a key
        if 'companyResolutionResult' in company_info:
            resolution_result = company_info['companyResolutionResult']
            return resolution_result.get("entityUrn", "").split(":")[-1]
    
    return None