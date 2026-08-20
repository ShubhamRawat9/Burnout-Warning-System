import logging
import os
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters,
    CallbackQueryHandler
)
from datetime import datetime, timedelta
import calendar

# ─────────────────────────────────────────
# CONFIG — loaded from environment variables.
# Set these in your shell, a local .env file (see .env.example),
# or your deployment platform's secrets manager. Never hardcode
# real values here or commit them to git.
# ─────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
SHEET_NAME = os.environ.get("SHEET_NAME", "Burnout Bot Data")
CREDENTIALS_FILE = os.environ.get(
    "GOOGLE_CREDENTIALS_PATH",
    os.path.join(os.path.dirname(__file__), "credentials.json")
)

if not BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is not set. Export it as an environment "
        "variable or add it to a local .env file before running the bot."
    )
# ─────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# GOOGLE SHEETS
# ─────────────────────────────────────────
def get_sheet():
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=scopes
    )
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME)

# ─────────────────────────────────────────
# CONVERSATION STATES
# ─────────────────────────────────────────
(REG_NAME, REG_ORG, REG_DESIGNATION,
 REG_EXPERIENCE, REG_WORK_MODE) = range(5)

(CHECKIN_CHOICE, CHECKIN_CALENDAR,
 CHECKIN_TASKS, CHECKIN_FOCUS,
 CHECKIN_DISTRACTIONS, CHECKIN_HOURS,
 CHECKIN_AFTER_HOURS, CHECKIN_FATIGUE) = range(5, 13)

LEAVE_DAYS = 13
LEAVE_CALENDAR = 14

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def is_registered(user_id):
    sheet = get_sheet()
    profiles = sheet.worksheet("User Profiles")
    all_ids = profiles.col_values(1)
    return str(user_id) in all_ids

def get_user_name(user_id):
    sheet = get_sheet()
    profiles = sheet.worksheet("User Profiles")
    all_ids = profiles.col_values(1)
    if str(user_id) in all_ids:
        row = all_ids.index(str(user_id)) + 1
        return profiles.cell(row, 2).value
    return "User"

def already_filled(user_id, date_str):
    try:
        sheet = get_sheet()
        daily = sheet.worksheet("Daily Data")
        all_values = daily.get_all_values()
        if len(all_values) <= 1:
            return False
        headers = all_values[0]
        for row in all_values[1:]:
            if len(row) >= 3:
                if str(row[0]) == str(user_id) and row[2] == date_str:
                    return True
        return False
    except Exception:
        return False

def save_daily_data(user_id, name, date_str, status,
                    tasks, focus, distractions,
                    hours, after_hours, fatigue):
    sheet = get_sheet()
    daily = sheet.worksheet("Daily Data")
    daily.append_row([
        str(user_id), name, date_str, status,
        tasks, focus, distractions,
        hours, after_hours, fatigue
    ])

def save_leave(user_id, name, date_str):
    sheet = get_sheet()
    daily = sheet.worksheet("Daily Data")
    daily.append_row([
        str(user_id), name, date_str, "LEAVE",
        "-", "-", "-", "-", "-", "-"
    ])

# ─────────────────────────────────────────
# CALENDAR BUILDER
# ─────────────────────────────────────────
def build_calendar(user_id, mode="checkin"):
    today = datetime.now().date()
    earliest = today - timedelta(days=21)  # 3 weeks back

    year = today.year
    month = today.month
    month_name = today.strftime("%B %Y")

    keyboard = []

    # Header
    keyboard.append([
        InlineKeyboardButton(
            f"📅 {month_name}", callback_data="ignore"
        )
    ])

    # Day headers
    keyboard.append([
        InlineKeyboardButton(d, callback_data="ignore")
        for d in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    ])

    # Get calendar for this month
    cal = calendar.monthcalendar(year, month)

    for week in cal:
        row = []
        for day in week:
            if day == 0:
                row.append(
                    InlineKeyboardButton(" ", callback_data="ignore")
                )
            else:
                date = datetime(year, month, day).date()
                date_str = date.strftime("%d/%m/%Y")

                # Check if date is selectable
                if date > today:
                    # Future date — not selectable
                    row.append(
                        InlineKeyboardButton(
                            f"·{day}·", callback_data="ignore"
                        )
                    )
                elif date < earliest:
                    # Too old — not selectable
                    row.append(
                        InlineKeyboardButton(
                            f"·{day}·", callback_data="ignore"
                        )
                    )
                elif already_filled(user_id, date_str):
                    # Already filled — show tick
                    row.append(
                        InlineKeyboardButton(
                            f"✓{day}", callback_data="ignore"
                        )
                    )
                else:
                    # Selectable date
                    row.append(
                        InlineKeyboardButton(
                            str(day),
                            callback_data=f"{mode}|{date_str}"
                        )
                    )
        keyboard.append(row)

    # Cancel button
    keyboard.append([
        InlineKeyboardButton("❌ Cancel", callback_data="cancel")
    ])

    return InlineKeyboardMarkup(keyboard)

# ─────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────
def main_menu():
    keyboard = [
        ["✅ Fill Today's Data"],
        ["📅 Fill Previous Days Data"],
        ["🏖️ Mark as On Leave"]
    ]
    return ReplyKeyboardMarkup(
        keyboard, resize_keyboard=True,
        one_time_keyboard=False
    )

def menu_message():
    return (
        "\n\n"
        "👇 Tap the square icon next to mic button "
        "if options are not visible"
    )

# ─────────────────────────────────────────
# START
# ─────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if is_registered(user_id):
        name = get_user_name(user_id)
        await update.message.reply_text(
            f"Welcome back, {name}! 👋\n"
            f"What would you like to do today?"
            f"{menu_message()}",
            reply_markup=main_menu()
        )
        return CHECKIN_CHOICE
    else:
        await update.message.reply_text(
            "Welcome to the Burnout Early Warning System! 👋\n\n"
            "This bot collects daily work behavior data\n"
            "for MCA research at UPES Dehradun.\n\n"
            "Your data is completely anonymous and secure.\n\n"
            "Let's start with a one-time registration.\n\n"
            "Question 1 of 5:\n"
            "What is your first name?"
        )
        return REG_NAME

# ─────────────────────────────────────────
# REGISTRATION
# ─────────────────────────────────────────
async def reg_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["name"] = update.message.text.strip()
    keyboard = [
        ["IT / Software", "Healthcare"],
        ["Finance / Banking", "Education"],
        ["Manufacturing", "Other"]
    ]
    await update.message.reply_text(
        f"Nice to meet you, {context.user_data['name']}! 😊\n\n"
        f"Question 2 of 5:\n"
        f"What type of organization do you work in?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True,
            one_time_keyboard=True
        )
    )
    return REG_ORG

async def reg_org(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["org"] = update.message.text.strip()
    keyboard = [
        ["Junior / Fresher", "Mid-level Employee"],
        ["Senior Employee", "Team Lead / Manager"],
        ["Other"]
    ]
    await update.message.reply_text(
        "Question 3 of 5:\n"
        "What is your current designation?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True,
            one_time_keyboard=True
        )
    )
    return REG_DESIGNATION

async def reg_designation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["designation"] = update.message.text.strip()
    keyboard = [
        ["Less than 1 year", "1-3 years"],
        ["3-5 years", "More than 5 years"]
    ]
    await update.message.reply_text(
        "Question 4 of 5:\n"
        "How many years of work experience do you have?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True,
            one_time_keyboard=True
        )
    )
    return REG_EXPERIENCE

async def reg_experience(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["experience"] = update.message.text.strip()
    keyboard = [["Full Remote", "Full Office", "Hybrid"]]
    await update.message.reply_text(
        "Question 5 of 5:\n"
        "Do you work remotely, from office, or hybrid?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True,
            one_time_keyboard=True
        )
    )
    return REG_WORK_MODE

async def reg_work_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["work_mode"] = update.message.text.strip()
    user_id = update.effective_user.id
    join_date = datetime.now().strftime("%d/%m/%Y")

    sheet = get_sheet()
    profiles = sheet.worksheet("User Profiles")
    profiles.append_row([
        str(user_id),
        context.user_data["name"],
        context.user_data["org"],
        context.user_data["designation"],
        context.user_data["experience"],
        context.user_data["work_mode"],
        join_date
    ])

    await update.message.reply_text(
        f"Registration complete! ✅\n\n"
        f"Welcome aboard, {context.user_data['name']}! 🎉\n\n"
        f"You will receive reminders at:\n"
        f"🔔 6:00 PM\n"
        f"🔔 8:00 PM\n"
        f"🔔 10:00 PM\n\n"
        f"(Only if you haven't filled your data)\n\n"
        f"It takes only 2 minutes per day."
        f"{menu_message()}",
        reply_markup=main_menu()
    )
    return CHECKIN_CHOICE

# ─────────────────────────────────────────
# CHECKIN CHOICE
# ─────────────────────────────────────────
async def checkin_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = update.message.text
    user_id = update.effective_user.id

    if choice == "✅ Fill Today's Data":
        today = datetime.now().strftime("%d/%m/%Y")
        context.user_data["checkin_date"] = today

        if already_filled(user_id, today):
            await update.message.reply_text(
                "✅ You have already filled today's data!\n\n"
                "See you tomorrow. Keep it up! 💪"
                f"{menu_message()}",
                reply_markup=main_menu()
            )
            return CHECKIN_CHOICE

        await update.message.reply_text(
            f"Filling data for today: {today} 📋\n\n"
            f"Question 1 of 6:\n"
            f"How many tasks did you complete today?\n"
            f"(Enter a number between 0 and 20)",
            reply_markup=ReplyKeyboardRemove()
        )
        return CHECKIN_TASKS

    elif choice == "📅 Fill Previous Days Data":
        await update.message.reply_text(
            "Select the date you want to fill 📅\n\n"
            "✅ Numbers = available to fill\n"
            "✓ = already filled\n"
            "· = not available",
            reply_markup=build_calendar(user_id, "checkin")
        )
        return CHECKIN_CALENDAR

    elif choice == "🏖️ Mark as On Leave":
        await update.message.reply_text(
            "How many days were you on leave?\n"
            "(Enter a number between 1 and 30)",
            reply_markup=ReplyKeyboardRemove()
        )
        return LEAVE_DAYS

    else:
        await update.message.reply_text(
            "Please choose one of the options below 👇\n\n"
            "If buttons are not visible — tap the square "
            "icon next to the mic button",
            reply_markup=main_menu()
        )
        return CHECKIN_CHOICE

# ─────────────────────────────────────────
# CALENDAR CALLBACK
# ─────────────────────────────────────────
async def calendar_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "ignore":
        return CHECKIN_CALENDAR

    if data == "cancel":
        await query.edit_message_text(
            "Cancelled. Use the menu to continue."
            f"{menu_message()}"
        )
        return CHECKIN_CHOICE

    if data.startswith("checkin|"):
        date_str = data.split("|")[1]
        context.user_data["checkin_date"] = date_str
        await query.edit_message_text(
            f"Filling data for: {date_str} 📅\n\n"
            f"Question 1 of 6:\n"
            f"How many tasks did you complete on this day?\n"
            f"(Enter a number between 0 and 20)"
        )
        return CHECKIN_TASKS

    if data.startswith("leave|"):
        date_str = data.split("|")[1]
        context.user_data["leave_start"] = date_str
        user_id = update.effective_user.id
        name = get_user_name(user_id)
        days = context.user_data.get("leave_days", 1)

        start_date = datetime.strptime(date_str, "%d/%m/%Y")
        saved_dates = []
        for i in range(days):
            leave_date_str = (
                start_date + timedelta(days=i)
            ).strftime("%d/%m/%Y")
            if not already_filled(user_id, leave_date_str):
                save_leave(user_id, name, leave_date_str)
                saved_dates.append(leave_date_str)

        await query.edit_message_text(
            f"✅ Leave marked successfully!\n\n"
            f"Dates marked as leave:\n" +
            "\n".join([f"🏖️ {d}" for d in saved_dates]) +
            f"\n\nHope you had a good rest! 😊"
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"What would you like to do next?{menu_message()}",
            reply_markup=main_menu()
        )
        return CHECKIN_CHOICE

# ─────────────────────────────────────────
# CHECKIN QUESTIONS
# ─────────────────────────────────────────
async def checkin_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tasks = int(update.message.text.strip())
        if not 0 <= tasks <= 20:
            raise ValueError
        context.user_data["tasks"] = tasks
        await update.message.reply_text(
            "Question 2 of 6:\n"
            "How focused were you throughout the day?\n"
            "(Enter a number from 1 to 10)\n\n"
            "1 = Could not focus at all\n"
            "10 = Completely focused all day"
        )
        return CHECKIN_FOCUS
    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a valid number between 0 and 20."
        )
        return CHECKIN_TASKS

async def checkin_focus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        focus = int(update.message.text.strip())
        if not 1 <= focus <= 10:
            raise ValueError
        context.user_data["focus"] = focus
        await update.message.reply_text(
            "Question 3 of 6:\n"
            "How many times were you distracted today?\n"
            "(Enter a number between 0 and 20)"
        )
        return CHECKIN_DISTRACTIONS
    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a number between 1 and 10."
        )
        return CHECKIN_FOCUS

async def checkin_distractions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        distractions = int(update.message.text.strip())
        if not 0 <= distractions <= 20:
            raise ValueError
        context.user_data["distractions"] = distractions
        await update.message.reply_text(
            "Question 4 of 6:\n"
            "How many hours did you work today?\n"
            "(Enter a number between 1 and 14)\n"
            "Example: 8 or 8.5"
        )
        return CHECKIN_HOURS
    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a number between 0 and 20."
        )
        return CHECKIN_DISTRACTIONS

async def checkin_hours(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        hours = float(update.message.text.strip())
        if not 1 <= hours <= 14:
            raise ValueError
        context.user_data["hours"] = hours
        keyboard = [["Yes", "No"]]
        await update.message.reply_text(
            "Question 5 of 6:\n"
            "Did you work after official office hours today?",
            reply_markup=ReplyKeyboardMarkup(
                keyboard, resize_keyboard=True,
                one_time_keyboard=True
            )
        )
        return CHECKIN_AFTER_HOURS
    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a number between 1 and 14.\n"
            "Example: 8 or 8.5"
        )
        return CHECKIN_HOURS

async def checkin_after_hours(update: Update, context: ContextTypes.DEFAULT_TYPE):
    after_hours = update.message.text.strip()
    if after_hours not in ["Yes", "No"]:
        keyboard = [["Yes", "No"]]
        await update.message.reply_text(
            "❌ Please select Yes or No.",
            reply_markup=ReplyKeyboardMarkup(
                keyboard, resize_keyboard=True,
                one_time_keyboard=True
            )
        )
        return CHECKIN_AFTER_HOURS
    context.user_data["after_hours"] = after_hours
    await update.message.reply_text(
        "Question 6 of 6:\n"
        "How fatigued do you feel right now?\n"
        "(Enter a number from 1 to 10)\n\n"
        "1 = Completely fresh\n"
        "10 = Completely exhausted",
        reply_markup=ReplyKeyboardRemove()
    )
    return CHECKIN_FATIGUE

async def checkin_fatigue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        fatigue = int(update.message.text.strip())
        if not 1 <= fatigue <= 10:
            raise ValueError
        context.user_data["fatigue"] = fatigue

        user_id = update.effective_user.id
        name = get_user_name(user_id)
        save_daily_data(
            user_id, name,
            context.user_data["checkin_date"],
            "FILLED",
            context.user_data["tasks"],
            context.user_data["focus"],
            context.user_data["distractions"],
            context.user_data["hours"],
            context.user_data["after_hours"],
            fatigue
        )

        await update.message.reply_text(
            "✅ Data saved successfully!\n\n"
            f"📅 Date         : {context.user_data['checkin_date']}\n"
            f"✅ Tasks        : {context.user_data['tasks']}\n"
            f"🎯 Focus        : {context.user_data['focus']}/10\n"
            f"😵 Distractions : {context.user_data['distractions']}\n"
            f"⏰ Hours        : {context.user_data['hours']}\n"
            f"🌙 After Hours  : {context.user_data['after_hours']}\n"
            f"😴 Fatigue      : {fatigue}/10\n\n"
            f"Thank you! See you tomorrow 💪"
            f"{menu_message()}",
            reply_markup=main_menu()
        )
        return CHECKIN_CHOICE

    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a number between 1 and 10."
        )
        return CHECKIN_FATIGUE

# ─────────────────────────────────────────
# LEAVE FLOW
# ─────────────────────────────────────────
async def leave_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        days = int(update.message.text.strip())
        if not 1 <= days <= 21:
            raise ValueError
        context.user_data["leave_days"] = days
        user_id = update.effective_user.id
        await update.message.reply_text(
            f"Select the start date of your leave 📅\n\n"
            f"✅ Numbers = available\n"
            f"✓ = already filled\n"
            f"· = not available",
            reply_markup=build_calendar(user_id, "leave")
        )
        return LEAVE_CALENDAR
    except ValueError:
        await update.message.reply_text(
            "❌ Please enter a valid number between 1 and 21."
        )
        return LEAVE_DAYS

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REG_NAME        : [MessageHandler(filters.TEXT & ~filters.COMMAND, reg_name)],
            REG_ORG         : [MessageHandler(filters.TEXT & ~filters.COMMAND, reg_org)],
            REG_DESIGNATION : [MessageHandler(filters.TEXT & ~filters.COMMAND, reg_designation)],
            REG_EXPERIENCE  : [MessageHandler(filters.TEXT & ~filters.COMMAND, reg_experience)],
            REG_WORK_MODE   : [MessageHandler(filters.TEXT & ~filters.COMMAND, reg_work_mode)],
            CHECKIN_CHOICE  : [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_choice)],
            CHECKIN_CALENDAR: [CallbackQueryHandler(calendar_callback)],
            CHECKIN_TASKS   : [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_tasks)],
            CHECKIN_FOCUS   : [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_focus)],
            CHECKIN_DISTRACTIONS: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_distractions)],
            CHECKIN_HOURS   : [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_hours)],
            CHECKIN_AFTER_HOURS: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_after_hours)],
            CHECKIN_FATIGUE : [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_fatigue)],
            LEAVE_DAYS      : [MessageHandler(filters.TEXT & ~filters.COMMAND, leave_days)],
            LEAVE_CALENDAR  : [CallbackQueryHandler(calendar_callback)],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    app.add_handler(conv_handler)

    print("✅ Bot is running...")
    print("Press Ctrl+C to stop")
    app.run_polling()

if __name__ == "__main__":
    main()