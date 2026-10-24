#!/usr/bin/env python3
"""Summarize completed V2 physical tasks from saved logs and post-run scores.

Only execution logs and evaluation files are read; nothing is sent to the
agent or back into a simulation. Pass the result JSON paths as arguments.
"""
import argparse
import json
import statistics
from pathlib import Path


def ratio(numerator,denominator):
    return None if denominator==0 else numerator/denominator


def distance_row(evidence):
    return next((row for row in evidence.get('constraints',[]) if row.get('requested_gap_m') is not None
                 and row.get('actual_gap_m') is not None),None)


def read_case(path):
    data=json.loads(path.read_text());result=data['result'];metrics=result.get('metrics',{})
    log=Path(result.get('log_path',''))/'events.jsonl'
    events=[json.loads(line) for line in log.open()] if log.is_file() else []
    validations=[event['data'] for event in events if event['type']=='final_validation'
                 and 'constraints' in event.get('data',{})]
    initial=distance_row(validations[0]) if validations else None
    final=distance_row(validations[-1]) if validations else None
    corrections=[event for event in events if event['type']=='placement_correction']
    pushes=0;push_successes=0
    for index,event in enumerate(events):
        if event['type']!='placement_correction' or event['data'].get('strategy')!='PUSH':continue
        pushes+=1
        following=next((later for later in events[index+1:] if later['type'] in
            {'final_validation','placement_correction'}),None)
        if following is not None and following['type']=='final_validation' and following['data'].get('accepted'):
            push_successes+=1
    truth=data.get('after_execution_ground_truth',{})
    truth_success=truth.get('within_tolerance',data.get('ground_truth_success'))
    truth_gap=truth.get('actual_gap_m',data.get('ground_truth_gap_m'))
    if truth_gap is None and final is not None:
        truth_gap=truth.get({'left_of':'left_edge_gap_m','right_of':'right_edge_gap_m'}.get(final.get('relation')))
    if truth_success is None and truth_gap is not None and final is not None:
        truth_success=abs(truth_gap-final['requested_gap_m'])<=final['tolerance_m']
    source_bounds=truth.get('source_bounds')
    table_edge=None
    if source_bounds is not None:
        task_file=Path(result['log_path'])/'task.json'
        if task_file.is_file():
            config=json.loads(task_file.read_text())['config']
            workspace=config['safety']['workspace']
            margin=config['spatial']['min_table_edge_margin_m']
            low,high=source_bounds
            table_edge=any(low[axis]<workspace[axis][0]+margin or high[axis]>workspace[axis][1]-margin
                           for axis in (0,1))
    return {'file':str(path),'task_id':result.get('task_id'),'agent_success':bool(result['success']),
            'ground_truth_success':truth_success,'relation_success':bool(validations and validations[-1].get('relation_satisfied')),
            'initial_placement_error_m':None if initial is None else abs(initial['actual_gap_m']-initial['requested_gap_m']),
            'final_visual_distance_error_m':None if final is None else abs(final['actual_gap_m']-final['requested_gap_m']),
            'final_ground_truth_distance_error_m':None if truth_gap is None or final is None else abs(truth_gap-final['requested_gap_m']),
            'correction_attempts':len(corrections),'correction_needed':bool(validations and not validations[0].get('accepted')),
            'push_attempts':pushes,'push_successes':push_successes,'regrasp_attempts':metrics.get('regrasp_attempts',0),
            'collision_count':max(data.get('collision_count',0),metrics.get('unexpected_contact_events',0)),
            'table_edge_violation':table_edge}


def summarize(cases):
    def nonnull(key):return [row[key] for row in cases if row[key] is not None]
    visual=nonnull('final_visual_distance_error_m');truth=nonnull('final_ground_truth_distance_error_m')
    initial=nonnull('initial_placement_error_m');corrected=[row for row in cases if row['correction_needed']]
    pushes=sum(row['push_attempts'] for row in cases)
    edge=nonnull('table_edge_violation')
    scored=[row for row in cases if row['ground_truth_success'] is not None]
    return {'tasks':len(cases),
            'spatial_relation_success_rate':ratio(sum(row['relation_success'] for row in cases),len(cases)),
            'mean_visual_distance_error_m':statistics.mean(visual) if visual else None,
            'median_visual_distance_error_m':statistics.median(visual) if visual else None,
            'mean_ground_truth_distance_error_m':statistics.mean(truth) if truth else None,
            'median_ground_truth_distance_error_m':statistics.median(truth) if truth else None,
            'mean_initial_placement_error_m':statistics.mean(initial) if initial else None,
            'correction_success_rate':ratio(sum(row['agent_success'] for row in corrected),len(corrected)),
            'push_success_rate':ratio(sum(row['push_successes'] for row in cases),pushes),
            'regrasp_rate':ratio(sum(row['regrasp_attempts']>0 for row in cases),len(cases)),
            'average_correction_attempts':statistics.mean(row['correction_attempts'] for row in cases),
            'collision_count':sum(row['collision_count'] for row in cases),
            'table_edge_violation_count':sum(edge) if edge else None,
            'table_edge_cases_scored':len(edge),
            'table_edge_definition':'post-run source bounds outside configured XY work surface with safety margin',
            'final_task_success_rate':ratio(sum(row['agent_success'] and row['ground_truth_success'] for row in scored),len(scored)),
            'ground_truth_cases_scored':len(scored)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases=[read_case(path) for path in args.results]
    report={'summary':summarize(cases),'cases':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
