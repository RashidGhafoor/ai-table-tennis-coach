"""
coach_agent.py

Takes evaluation reports and generates human-friendly coaching suggestions and a short practice plan.
This module includes a function `generate_coaching_plan` which returns textual instructions and drill lists.
"""

def generate_coaching_plan(evaluations, user_profile=None):
    plan = {'summary': '', 'drills': [], 'schedule': []}
    # Aggregate common issues
    problem_counts = {}
    for e in evaluations:
        for issue in e.get('issues', []):
            problem_counts[issue] = problem_counts.get(issue, 0) + 1
    # Build summary
    if not problem_counts:
        plan['summary'] = 'Good technique overall. Focus on consistency and footwork.'
    else:
        issues_sorted = sorted(problem_counts.items(), key=lambda x: -x[1])
        plan['summary'] = 'Common issues: ' + ', '.join([f"{i[0]} (x{i[1]})" for i in issues_sorted])
    # Simple drill generation
    if any('Elbow' in k or 'elbow' in k for k in problem_counts):
        plan['drills'].append({'name': 'Elbow alignment drill', 'description': 'Practice slow-motion forehands focusing on keeping elbow below shoulder. 3 sets of 10.'})
    if any('Racket angle' in k or 'racket' in k.lower() for k in problem_counts):
        plan['drills'].append({'name': 'Open-face drill', 'description': 'Practice pushing the ball with a more open racket face. Try 5 sets of 15 feeds.'})
    # Schedule: 2-week starter
    plan['schedule'] = [
        {'day': 1, 'focus': 'Drills: ' + (plan['drills'][0]['name'] if plan['drills'] else 'Consistency drills')},
        {'day': 3, 'focus': 'Multi-ball: work on rhythm and contact point'},
        {'day': 5, 'focus': 'Match play focusing on implementing technique'},
    ]
    return plan

if __name__ == '__main__':
    import json,sys
    data = json.load(open(sys.argv[1]))
    report = generate_coaching_plan(data)
    print(report)
