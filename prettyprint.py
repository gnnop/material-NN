
from termcolor import colored



def prettyPrint(data, indent=0, indent_step=2, color_index=0, only_show_amount=2):
    """
    Pretty prints a complicated data structure with nested iterables, colorizing each layer and
    optionally abridging long lists by showing only a specified number of elements from the start and end.
    """
    colors = ['yellow', 'red', 'blue', 'green', 'magenta', 'cyan', 'white']

    def abridge_sequence(sequence, only_show_amount):
        if only_show_amount < 1 or len(sequence) <= only_show_amount * 2:
            return sequence
        return sequence[:only_show_amount] + ['...'] + sequence[-only_show_amount:]

    def print_item(item, indent, color_index, is_last_item=False):
        current_color = colors[color_index % len(colors)]

        if isinstance(item, (list, tuple, set)):
            item_repr = list(item) if not isinstance(item, list) else item
            if only_show_amount > -1 and len(item_repr) > only_show_amount * 2:
                item_repr = abridge_sequence(item_repr, only_show_amount)
            if all(not isinstance(sub_item, (list, tuple, set, dict)) for sub_item in item_repr):
                items_str = ', '.join(repr(sub_item) for sub_item in item_repr)
                print(colored((' ' * indent) + f'{type(item).__name__} [{len(item)}]: {items_str}', current_color))
            else:
                print(colored((' ' * indent) + f'{type(item).__name__} [{len(item)}]:', current_color))
                for sub_item in item_repr:
                    if sub_item == '...':
                        print(colored((' ' * (indent + indent_step)) + '...', current_color))
                    else:
                        print_item(sub_item, indent + indent_step, color_index + 1, is_last_item=True)
        elif isinstance(item, dict):
            print(colored((' ' * indent) + f'dict [{len(item)}]:', current_color))
            for k, v in item.items():
                print(colored((' ' * (indent + indent_step)) + f'{k}:', current_color), end='')
                if isinstance(v, (list, tuple, set, dict)):
                    print()
                    print_item(v, indent + indent_step * 2, color_index + 1, is_last_item=True)
                else:
                    print(colored(f' {v}', colors[(color_index + 1) % len(colors)]))
        else:
            if is_last_item:
                print(colored((' ' * indent) + f'{item}', current_color))
            else:
                print(colored((' ' * indent) + f'- {item}', current_color))
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(colored(f'{key} [{len(data)}]:', colors[color_index % len(colors)]), end='')
            if isinstance(value, (list, tuple, set, dict)):
                print()
                print_item(value, indent + indent_step, color_index + 1)
            else:
                print(colored(f' {value}', colors[(color_index + 1) % len(colors)]))
    else:
        print_item(data, 0, color_index)
# end of pretty_print


