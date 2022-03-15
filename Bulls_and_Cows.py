import random


def check_number(number):
    if len(number) != 4:
        print('e1')
        return False
    if not number.isdigit():
        print('e2')
        return False
    if len(set(number)) != 4:
        print('e3')
        return False

    else:
        return True


def count_bc(guess, answer):
    if check_number(guess):
        b = 0
        c = 0
        for i, l in enumerate(guess):
            if l in answer:
                if answer[i] == l:
                    b += 1
                else:
                    c += 1
        return b, c

    else:
        print('Error')


def start_the_game():
    global answer
    answer = ''
    while not check_number(answer):
        answer = str(random.randint(123, 9877))


def reveal_answer():
    global answer
    print(f'{answer}')


stop = False
start = False
while not stop:
    inp = input()
    if inp == 'start':
        # print('START')
        start_the_game()
        tries = 0
        start = True
        # print(f'{answer}')
    elif inp == 'reveal':
        # print('REVEAL')
        if not start:
            print(f"The game hasn't started yet!")
        else:
            reveal_answer()

    elif inp == 'stop':
        # print('STOP')
        stop = True
    else:
        # print('ELSE')
        if not start:
            print(f"The game hasn't started yet!")
        else:
            guess = str(inp)
            b, c = count_bc(guess, answer)
            if b == 4:
                print(f'You won! It took you {tries} tries')
            else:
                print(f'bulls {b} cows {c}')
                tries+=1