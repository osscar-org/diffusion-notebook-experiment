import json
from collections import defaultdict
import pylab as plt

ADJUST_TEXT = False
events_by_uid = defaultdict(list)

rows = [
    "IntSlider-Number of points $N$",
    "IntSlider-Time step # $(t)$",
    "IntSlider-Number of time steps $t$",
    "FloatSlider-Step size $l$",
    "FloatSlider-$p_x$",
    "LoggingPlay-",
    "Button-Simulate",
    "Checkbox-Show trajectory of one particle"
]

def parse_lines(lines):
    parsed = []

    for line in lines:
        pieces = line.split()
        if len(pieces) >= 8:
            # we need at least this number of pieces
            month, day, time, timestamp, server, logger, level = pieces[:7] 
            txt_data = " ".join(pieces[7:])
            data = json.loads(txt_data)
            parsed.append({
                "month": month,
                "day": day,
                "time": time,
                "timestamp": float(timestamp),
                "server": server,
                "logger": logger,
                "level": level,
                "data": data
            })
        else:
            print("# skipping line")
    return parsed

if __name__ == "__main__":
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "OSSCAR_testing.log"
    with open(filename) as fhandle:
        lines = fhandle.readlines()
    
    parsed_events = parse_lines(lines)
    # Filter
    filtered_events = [event for event in parsed_events if event['logger'] in ["diffusion_module", "cedeLogger"]]

    for event in filtered_events:
        events_by_uid[event['data']['kid']].append(event)

    print(f"### {len(filtered_events)} logged events")
    # In principle we could start the time not from the first action, but from the 
    # execution of the cells - this is also in the logs
    for uid, events in events_by_uid.items():
        first_timestamp = None
        
        fig = plt.figure(figsize=(16,6))
        ax = plt.subplot(111)
        print(f"- {len(events)} events for {uid}")
        last_play_time = None
        last_reset_time = 0
        texts = []
        for event in events:
            print(event)
            if 'load' in event['data']['raw_event']:
                if first_timestamp is None:
                    first_timestamp = event['timestamp']
                    continue
                else:
                    raise ValueError("Error! Loaded twice?")
            name = f" '{event['data']['which']}'" if event['data']['which'] else ""
            value_description = f" from value {event['data']['from_value']} to {event['data']['to_value']}" if 'from_value' in event['data'] else ""
            if first_timestamp is None:
                raise ValueError("Data for a kernel without a 'start' event!")
                #first_timestamp = event['timestamp']
            time = event['timestamp'] - first_timestamp

            print(f"  {int(time):5d}s: {event['data']['what']}{name}{value_description}")
            # f'{event['data']['where']} on '
            row_idx = rows.index(f"{event['data']['what']}-{event['data']['which']}")
            if event['data']['what'] == "LoggingPlay":
                if event['data']['to_value'] is True:
                    if last_play_time is not None:
                        raise ("Two play actions in a row!")
                    last_play_time = time
                    plt.plot([last_play_time], [row_idx], '.g')
                else:
                    if last_play_time is None:
                        raise ("Stop action without a play first!")

                    plt.plot([last_play_time, time], [row_idx, row_idx], '-g')
                    plt.plot([time], [row_idx], '.r')
                    last_play_time = None
            elif event['data']['what'] == "Button":
                plt.plot([time], [row_idx], '.', color='blue')
                if event['data']['which'] == "Simulate":
                    ax.fill_between(
                        [last_reset_time, time], 0, 1,
                        color='gray', alpha=0.5,
                        transform=ax.get_xaxis_transform()
                    )
            elif event['data']['what'] == "Checkbox":
                if event['data']['to_value'] is True:
                    plt.plot([time], [row_idx], '.', color='green')
                else:
                    plt.plot([time], [row_idx], '.', color='red')
            elif event['data']['what'] in ["IntSlider", "FloatSlider"]:
                if event['data']['to_value'] > event['data']['from_value']:
                    plt.plot([time], [row_idx], '.', color='green')
                else:
                    plt.plot([time], [row_idx], '.', color='red')
                # will show the text and append it
                if isinstance(event['data']['to_value'], float):
                    str_val = f"{event['data']['to_value']:.2f}"
                else:
                    str_val = str(event['data']['to_value'])
                texts.append(plt.text(time, row_idx, str_val))
            else:
                raise ValueError("Unknown event")
                #plt.plot([time], [row_idx], '.', color='orange')

            # Show bar when no simulation has been computed - we do it
            # when there is an event that resets the simulutions
            if (event['data']['what'], event['data']['which']) in [
                ('IntSlider', 'Number of points $N$'),
                ('IntSlider', 'Number of time steps $t$'),
                ('FloatSlider', 'Step size $l$'),
                ('FloatSlider', '$p_x$'),
            ]:
                last_reset_time = time

        ax.set_yticks(range(len(rows)), rows)
        plt.subplots_adjust(left=0.2, right=0.99)
        plt.xlabel("Time from first event (s)")
        plt.xlim(-1, events[-1]['timestamp'] - first_timestamp+1)
        plt.title(f"Log for user {uid}")
        # (Optionally) Fix the text location
        if ADJUST_TEXT:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

        plt.show()