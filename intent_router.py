from intent_detector import route_query
from intent_types import answer_single,answer_multi,answer_temporal,answer_analytical

def answer(user_question,data,client,model) -> str:
    route = route_query(user_question, client)
    intent = route["route"]
    
    try:
        if intent == "analytical":
            return answer_analytical(user_question, route, data, client)

        elif intent == "temporal":
            return answer_temporal(user_question, route, data, client)

        elif intent == "multi":
            return answer_multi(user_question, route, data, client)

        elif intent == "single":
            return answer_single(user_question, route, data, client) #  answer_single(user_question, route, data, client)
    except:
        return {'response': False, 'text': '', 'questions': []}