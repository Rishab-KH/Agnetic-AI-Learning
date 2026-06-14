import os
import sys
from pathlib import Path
import time

from typing import Any
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


from utils.log_config import log_tool_call
from utils.tool_response import success_response, error_response
from utils.validators import validate_choice, validate_positive_integer

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

SEVERITIES = ['mild', 'moderate', 'severe']
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
TIMEOUT_SECONDS = 8

# System Prompt for Agent

SYSTEM_PROMPT = '''
    You are a healthcare information assistant. You have access to two tools:
    1. estimate_visit_priority — given a symptom, severity (mild/moderate/severe), and age,
    returns a recommended visit priority (routine/soon/urgent/emergency). Use this whenever
    a user describes symptoms, even briefly. If severity or age is missing or unclear, ask
    the user for it before calling the tool — do not guess.

    2. lookup_drug_info — given a drug name, returns FDA label information (what it's used
    for and key warnings). Use this when a user asks about a medication by name.

    Guidelines:
    - If a user describes symptoms AND mentions a medication, use both tools as relevant.
    - Never provide a diagnosis. Only relate tool output back to the user in plain language.
    - If estimate_visit_priority returns "emergency" or "urgent", clearly and prominently
    recommend the user seek care promptly (e.g. ER or urgent care) — do not soften this.
    - If a tool returns an error, explain the issue in plain language and ask the user for
    corrected input (e.g. if severity is invalid, list the valid options).
    - Always end medication or symptom responses with a brief reminder that this is not a
    substitute for professional medical advice, and a doctor or pharmacist should be
    consulted for personal medical decisions.
    - Be concise, calm, and clear — avoid alarming language unless urgency genuinely warrants it.
'''

# map when to visit the hospital 3 * 2 combinations
PRIORITY_MATRIX = {
    ('mild', False): "routine",
    ('mild', True): "routine",
    ('moderate', False): "soon",
    ('moderate', True): "urgent",
    ('severe', False): "urgent",
    ('severe', True): "emergency",
}

# --------------------------------Tools----------------------------------------

@tool
def estimate_visit_priority(
    symptom: str,
    severity: str,
    age: int,
    has_risk_factor: bool = False
    ) -> dict:
    """
        Estimate how soon a patient should visit a hospital based on the reported symptom, severity of the symptom, age, and whether they have a known risk factor (e.g. chronic
        condition, immunocompromised).
        Use this tool automatically when the symptom, severity, age and risk factor is available. It's a deterministic lookup tool, not user invoked.
        Args:
             symptom (str): Free-text symptom description (used only for the reason field).
            severity (str): One of "mild", "moderate", "severe".
            age (int): Patient age in years. Must be > 0. Ages 65+ are treated
                as an elevated risk factor regardless of has_risk_factor.
            has_risk_factor (bool): True if the patient has an additional elevated
                risk profile (e.g. chronic condition, immunocompromised). Defaults to False.
 
    Returns:
            Success envelope with:
                - priority (str): "routine", "soon", "urgent", or "emergency"
                - reason (str): Human-readable explanation
            Error envelope (invalid_input) if severity is not recognized or age <= 0.
    """

    start_time = time.perf_counter()
    input_data = {
        "symptom": symptom,
        "severity": severity,
        "age": age,
        "has_risk_factor": has_risk_factor
    }

    value, options, name = severity, SEVERITIES, "severity"
    ok, message = validate_choice(value, options, name)

    if not ok:
        latency = (time.perf_counter() - start_time) * 1000
        log_tool_call(
            tool_name='estimate_visit_priority',
            input_data=input_data,
            status="error",
            latency_ms=latency,
            error_type="invalid input"
        )
        return error_response(
            error_type=f"invalid {name} choice input",
            message=message)
    
    val, field_name = age, "age"
    ok, message = validate_positive_integer(val, field_name)
    if not ok:
        latency = (time.perf_counter() - start_time) * 1000
        log_tool_call(
            tool_name='estimate_visit_priority',
            input_data=input_data,
            status="error",
            latency_ms=latency,
            error_type=f"invalid input"
        )

        return error_response(
            error_type=f"invalid {field_name} input",
            message=message)

    # Happy Path

    effective_risk_factor = has_risk_factor or age >= 65
    severity_type = severity.strip().lower()
    priority = PRIORITY_MATRIX[(severity_type, effective_risk_factor)]
    risk_note = "with an elevated risk factor" if effective_risk_factor else "with no elevated risk factors"
    reason = f"Reported '{symptom}' as {severity_type} severity, patient age {age}, and risk note {risk_note}"
    
    latency = (time.perf_counter() - start_time) * 1000
    log_tool_call(
        tool_name='estimate_visit_priority',
        input_data=input_data,
        status='success',
        latency_ms=latency
    )
    return success_response({'priority': priority,
                             'reason': reason},
                             source='estimate_visit_priority local tool')


@tool
def drug_lookup_info(drug_name: str) -> dict[str, Any]:
    """
    Fetches FDA label information for a drug: what it's used for and warnings.
    Use this tool when the user asks what a medication is for, how it's used, or what it's warnings or side-effects are.

    Args:
        drug_name (str): Brand or generic drug name (e.g. "Metaformin", "Tylenol").

    Return:
        - brand_name (str)
        - generic_name (str)
        - indicatons (str): used for text
        - warnings (str): warnings text
    """
    
    start = time.perf_counter()
    input_data = {"drug_name":drug_name}
    if not isinstance(drug_name, str):
        latency=round(time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name=drug_lookup_info,
            input_data=input_data,
            status="error",
            latency_ms=latency,
            error_type=f"invalid_input"
        )
        return error_response("invalid_input", f"Invalid Input for {drug_name} of type {type(drug_name)} expected type str")
    try:
        resp = requests.get(
            OPENFDA_URL,
            params={
                "search": f"openfda.brand_name: '{drug_name}' OR openfda.generic_name: '{drug_name}'",
                "limit": 1
            },
            timeout=TIMEOUT_SECONDS
        )
    except requests.Timeout:
        latency = (time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name='drug_lookup_info',
            input_data=input_data,
            status='error',
            latency_ms=latency,
            error_type="timeout"
        )

        return error_response(
            error_type="timeout",
            message="The drug lookup service took too long to respond"
        )
    except requests.RequestException: # we get request exception for upstream most likely the API error
        latency = (time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name='drug_lookup_info',
            input_data=input_data,
            status='error',
            latency_ms=latency,
            error_type="upstream_error"
        )

        return error_response(
            error_type="upstream_error",
            message="Couldn't reach the drug lookup service right now"
        )
    
    if resp.status_code == 404:
        latency = (time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name='drug_lookup_info',
            input_data=input_data,
            status='error',
            latency_ms=latency,
            error_type="not_found"
        )
        return error_response(
            "not_found",
            f"No drug found with name {drug_name}.",
        )
    
    if resp.status_code != 200:
        latency = (time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name='drug_lookup_info',
            input_data=input_data,
            status='error',
            latency_ms=latency,
            error_type="upstream_error"
        )
        return error_response(
            error_type="upstream_error",
            message="The drug lookup service returned an unexpected response"
        )
    
    try:
        raw = resp.json()
    except ValueError:
        latency = (time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name='drug_lookup_info',
            input_data=input_data,
            status='error',
            latency_ms=latency,
            error_type='upstream_error'
        )
        return error_response(
            error_type="upstream_error",
            message="The drug lookup service returned data we couldn't read."
        )
    results = raw.get("results")
    if not results:
        latency = (time.perf_counter() - start) * 1000
        log_tool_call(
            tool_name='drug_lookup_info',
            input_data=input_data,
            status='error',
            latency_ms=latency,
            error_type='not_found'
        )
        return error_response(
        error_type="not_found",
        message=f"No FDA label found for '{drug_name}'."
        )

    record = results[0]  
    openfda = record.get("openfda", {})
    warnings_text = (
    record.get("warnings")
    or record.get("warnings_and_precautions")
    or record.get("boxed_warning")
    or ["N/A"]
    )
    
    normalized = {
    "brand_name":   openfda.get("brand_name", ["N/A"])[0],
    "generic_name": openfda.get("generic_name", ["N/A"])[0],
    "indications":  record.get("indications_and_usage", ["N/A"])[0],
    "warnings":     warnings_text[0]
    }

    latency = (time.perf_counter() - start) * 1000
    log_tool_call(
        "drug_lookup_info",
        input_data,
        status="success",
        latency_ms=latency,
    )
    return success_response(normalized, source="openfda_api")


# --------------------------------Agent----------------------------------------

TOOLS = [estimate_visit_priority, drug_lookup_info]

def run_agent(agent: ChatOpenAI, user_input: str) -> None:
    '''
        Runs the agent based on the user input about visit priority to a hospital or drug information lookup
    '''

    print("\n" + "=" * 60)
    print(f"USER: {user_input}")
    print("=" * 60)

    results = agent.invoke({
        'messages': [HumanMessage(content=user_input)]
    })

    for msg in results['messages']:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                print(f"Agent wants to call tool: {call['name']}({call['args']})")
            if isinstance(msg, ToolMessage):
                print(f"Tool results {msg.content}")
    
    final = results['messages'][-1]
    if getattr(final, "content", None):
        print(f"Agent Response: {final.content}")

 
def main():
    result = estimate_visit_priority.invoke({
                "symptom": "chest tightness",
                "severity": "severe",
                "age": 70,
                "has_risk_factor": False,
            })
    print("Local success:", result)
    print("=" * 60)

    result = estimate_visit_priority.invoke({
                "symptom": "fever",
                "severity": "moderate",
                "age": -5,
                "has_risk_factor": False,
            })
    print(f"Local error handled safely {result}")
    print("=" * 60)
 
    # API tool tests
    result = drug_lookup_info.invoke({"drug_name": "Metformin"})
    print(f"Successfull API tool call {result}")
    print("=" * 60)

    result = drug_lookup_info.invoke({"drug_name": "NotARealDrugXYZ"})
    print(f"Failed API tool call handled safely {result}")

    ## Agent based response
    # agent = create_agent(
    #         model="gpt-4o-mini",
    #         tools=TOOLS,
    #         system_prompt=SYSTEM_PROMPT,
    # )


    # # estimate_visit_priority — normal cases
    # run_agent(
    #     agent,
    #     "I'm 70 years old and have severe chest tightness. What should I do?",
    # )
    # run_agent(
    #     agent,
    #     "My daughter is 8 and has a mild sore throat with no other health issues. "
    #     "How soon should we see a doctor?",
    # )

    # # estimate_visit_priority — invalid/edge inputs
    # run_agent(
    #     agent,
    #     "I'm -5 years old and have moderate stomach pain. How urgent is this?",
    # )
    # run_agent(
    #     agent,
    #     "I have an extreme fever, I'm 30, what priority is this?",
    # )

    # # lookup_drug_info — normal cases
    # run_agent(agent, "What is Metformin used for, and are there any warnings?")
    # run_agent(agent, "Can you tell me about Lisinopril?")

    # # lookup_drug_info — not found / unusual input
    # run_agent(agent, "What is NotARealDrugXYZ used for?")

    # # combined — both tools in one query
    # run_agent(
    #     agent,
    #     "I'm 68, I take Metformin, and I've had severe abdominal pain since this "
    #     "morning. Should I be worried, and is this a known side effect of my medication?",
    # )

        


if __name__ == "__main__":
    # Local tool tests
    main()

    




