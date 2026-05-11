ZERO_SHOT_PROMPT = (
    "USER: <image>\n"
    "You are an experienced radiologist reviewing a chest X-ray. "
    "Write a structured radiology report with two sections:\n"
    "1. Findings: describe all observable findings using precise clinical terminology.\n"
    "2. Impression: provide a concise one- to two-sentence clinical summary.\n"
    "Do not speculate beyond what is visible.\n"
    "ASSISTANT:"
)

TRAIN_PROMPT = (
    "USER: <image>\n"
    "Write the radiology report (Findings and Impression) for this chest X-ray.\n"
    "ASSISTANT:"
)

def build_zero_shot(image_token="<image>"):
    return ZERO_SHOT_PROMPT.replace("<image>", image_token)

def build_train(findings, impression, image_token="<image>"):
    prompt = TRAIN_PROMPT.replace("<image>", image_token)
    target = f"Findings: {findings.strip()}\nImpression: {impression.strip()}"
    return prompt + " " + target

def format_target(findings, impression):
    return f"Findings: {findings.strip()}\nImpression: {impression.strip()}"
