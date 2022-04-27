import logging
import pickle as pkl
import random
from telegram import *
from telegram.ext import *
from requests import *


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    buttons = [[KeyboardButton('Train'), KeyboardButton('Finished')],
               [KeyboardButton('TestImage'), KeyboardButton('Timer'), KeyboardButton('Help')]]
    context.bot.send_message(chat_id=update.effective_chat.id, text="Create or start a game with a code!",
                             reply_markup=ReplyKeyboardMarkup(buttons))


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('TODO help message', parse_mode=ParseMode.MARKDOWN_V2)

#
# def join(update, context):
#     """When joined"""
#     try:
#         code = int(context.args[0])
#         if code in my_games.game_codes:
#             game = my_games.games[code]
#             if game.is_started:
#                 update.message.reply_text(f'The game has already started!')
#             else:
#                 p = player(update)
#                 my_games.new_player(p, code)
#                 update.message.reply_text(f'Joined room with code {code}')
#                 update.message.reply_text(f'You are player number {len(my_games.games[code].players)}')
#                 print(f'{p.id} joined game {code} with number {len(my_games.games[code].players)}')
#                 # print(len(my_games.games[code].players))
#                 # print(my_games.games[code].players[0].id)
#                 # print(my_games.games[code].players[1].id)
#                 # print(my_games.games[code].code)
#         else:
#             update.message.reply_text(f'No room with code {code}')
#     except (IndexError, ValueError):
#         update.message.reply_text(f'Some problem n1!')
#     # update.message.reply_text('Joined !')
#
#
# def create(update, context):
#     """Send a message when the command /help is issued."""
#     try:
#         # code = int(context.args[0])
#         code = 212
#         the_dictionary = 0
#         try:
#             the_dictionary = int(context.args[1])
#         except:
#             update.message.reply_text(f'Dictionary 1 is chosen')
#         if code in my_games.game_codes:
#             update.message.reply_text(f'Room with code {code} already exists')
#         else:
#             print('not in game codes')
#             if the_dictionary:
#                 print('dict on')
#                 the_game = game(code, dictionary=the_dictionary)
#             else:
#                 the_game = game(code)
#
#             p = player(update)
#             the_game.players.append(p)
#             my_games.add_game(the_game)
#             # print('Choose n2')
#             update.message.reply_text(f'Choose dictionary')
#             # print(len(the_game.players))
#             # print(the_game.players[0].id)
#             # print(the_game.code)
#             # TODO choose dictionary
#     except (IndexError, ValueError):
#         update.message.reply_text(f'Some problem n1!')
#     # update.message.reply_text('Joined !')
#
#
# def get_game(update, running_games=my_games):
#     try:
#         user_id = update.message.from_user['id']
#     except:
#         user_id = update.callback_query.from_user['id']
#     if user_id in running_games.user_game:
#         game = running_games.user_game[user_id]
#     else:
#         game = list(running_games.games.values())[0]
#     #TODO fix
#     return game
#
#
# def round(update, context):
#     """Make round"""
#     # print(1)
#     # user_id = update.message.from_user['id']
#     # print(2)
#     game = get_game(update)
#     # print(3)
#     game.make_round()
#     # print(4)
#     for player in game.players:
#         id = player.id
#         player.state = 'words'
#         my_words = game.round_words[id]
#         # buttons = [[KeyboardButton(my_words[0])], [KeyboardButton(my_words[1])]]
#         # update.message.reply_text('Choose', reply_markup=ReplyKeyboardMarkup(buttons))
#         buttons = [[InlineKeyboardButton(my_words[0], callback_data="first")],
#                    [InlineKeyboardButton(my_words[1], callback_data="second")]]
#         # print('context')
#         # print(context)
#         # print('bot')
#         # print(context.bot)
#         context.bot.send_message(chat_id=id, reply_markup=InlineKeyboardMarkup(buttons),
#                                  text="Choose a word to draw")
#
#
# def picked_n(update, context, n: (1, 2)):
#     # print(21)
#     user_id = update.callback_query.from_user['id']
#     # print(user_id)
#     # print(22)
#     game = get_game(update)
#     # print(23)
#     player = game.get_player(user_id)
#     # print(24)
#     player.state = 'picked'
#     # print(25)
#     my_pick = game.round_words[user_id][n - 1]
#     if not game.hide:
#         context.bot.send_message(chat_id=user_id,
#                                  text=f'{game.round_words[user_id][n - 1]} chosen')
#     # print(26)
#     game.chosen_words[user_id] = my_pick
#     if len(game.chosen_words) == len(game.players):
#         reveal_words = list(game.chosen_words.values()) + list(game.round_words['leftovers'])
#         random.shuffle(reveal_words)
#         game.reveal_words = reveal_words
#     # print(27)
#
#
# def picked_first(update, context):
#     picked_n(update, context, 1)
#
#
# def picked_second(update, context):
#     picked_n(update, context, 2)
#
#
# def hide(update, context):
#     game = get_game(update)
#     game.hide = not game.hide
#     update.message.reply_text('Words are hidden now' if game.hide else 'Words are shown now')
#
#
# def change_dice(update, context):
#     game = get_game(update)
#     game.dice = not game.dice
#     update.message.reply_text('Dice are rolled now' if game.hide else 'Find your own dice!')
#
#
# def queryHandler(update: Update, context: CallbackContext):
#     # print(11)
#     query = update.callback_query.data
#
#     # print(12)
#     update.callback_query.answer()
#     # print(13)
#     if "first" in query:
#         # print(14)
#         picked_first(update, context)
#         # print(15)
#         update.callback_query.edit_message_text(text=f"Selected option: {query}")
#     elif "second" in query:
#         # print(16)
#         picked_second(update, context)
#         # print(17)
#         update.callback_query.edit_message_text(text=f"Selected option: {query}")
#     elif "finish game" in query:
#         update.callback_query.edit_message_text(text=f"Game has finished! To start a new game write /create [code]")
#     elif "cancel" in query:
#         update.callback_query.edit_message_text(text=f"Great! Continue playing!")
#     # message_id = query.inline_message_id
#

# def choose(update, context):
#     """choose"""
#     # user = update.message.from_user['id']
#     # game = get_game(user)
#     # my_words = game.round_words[user]
#     # if my_words[0] in update.message.text:
#     #     update.message.reply_text('picked first')
#     # if my_words[1] in update.message.text:
#     #     update.message.reply_text('picked second')
#     # chosen_word = update.message.text.lower()
#     # game = get_game(update, my_games)
#     # game.add_word(chosen_word)
#     update.message.reply_text('unknown n1')

def handle_message(update, context):
    if 'Help' in update.message.text:
        help(update, context)

    if 'TestImage' in update.message.text:
        context.bot.send_photo(update.message.from_user.id,
                               photo=open('/Users/balepka/Downloads/rnn_torchviz1.png', 'rb'))

    elif 'Timer' in update.message.text:
        update.message.reply_text("Timer ping")

    elif 'Train' in update.message.text:
        update.message.reply_text("Train ping")


    elif 'Finished' in update.message.text:
        buttons = [[InlineKeyboardButton('Stop', callback_data="stop_train")],
                   [InlineKeyboardButton('Train_next', callback_data="train_next")]]
        context.bot.send_message(chat_id=update.message.from_user.id, reply_markup=InlineKeyboardMarkup(buttons),
                                 text="Now what?")




def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


# def game_stats(update, context):
#     game = get_game(update)
#     print(f'players: {list(p.id for p in game.players)}')
#     print(f'round words: {game.round_words}')
#     print(f'chosen words: {game.chosen_words}')
#     print(f'reveal words: {game.reveal_words}')
#
#
# def reveal(update, context):
#     game = get_game(update)
#     game.reveal(context)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/token.txt') as f:
        token = f.read()
    updater = Updater(token, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    # dp.add_handler(CommandHandler("join", join))
    # dp.add_handler(CommandHandler("create", create))
    # dp.add_handler(CommandHandler("round", round))
    # dp.add_handler(CommandHandler("first", picked_first))
    # dp.add_handler(CommandHandler("second", picked_second))
    # dp.add_handler(CommandHandler("game", game_stats))
    # dp.add_handler(CommandHandler("reveal", reveal))
    # dp.add_handler(CommandHandler("hide", hide))
    # dp.add_handler(CommandHandler("dice", hide))

    # dp.add_handler(CallbackQueryHandler(queryHandler))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, handle_message))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
