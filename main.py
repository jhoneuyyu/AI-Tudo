from fastmcp import FastMCP


import sqlite3

import logging

mcp=FastMCP("Tudo manager")



# Configuration
DB="tudo.sqlite3"
LOG_FILE="event_log.log"


# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def init_db():
    """Initialize the database of todos and events"""
    conn=sqlite3.connect(DB)
    c=conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT NOT NULL,
        description TEXT,
        due_date TEXT CHECK(
            due_date IS NULL OR due_date GLOB '____-__-__T__:__:*'
        ),
        completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0,1)),
        priority INTEGER NOT NULL DEFAULT 0 CHECK(priority >= 0 AND priority <= 5),
        category TEXT NOT NULL DEFAULT 'General',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TEXT DEFAULT NULL,
        deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0,1))
    )""")
    conn.commit()
    conn.close()

   

def trigger_task_event(action: str, task_info: str, message: str = "", status: str = "info"):
    logging.info(f"EVENT FIRED: {action} task={task_info} message={message} status={status}")


    







  

@mcp.tool(name="addtask", description="Add a new task to the todo list")
def add_task(task: str, description: str = "", due_date: str = None, priority: int = 0, category: str = "General"):
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("""INSERT INTO todos(task, description, due_date, priority, category) VALUES(?, ?, ?, ?, ?)""", 
                  (task, description, due_date, priority, category))
        conn.commit()
        conn.close()
        trigger_task_event("add_task", task, "Task added successfully", "success")
        return "Task added successfully"
    except Exception as e:
        trigger_task_event("add_task", task, str(e), "error")
        raise e


@mcp.tool(name="listtasks", description="Get all tasks from the todo list")
def list_tasks(completed: int = 0):
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        if completed:
            c.execute("""SELECT * FROM todos WHERE completed=?""", (completed,))
        else:
            c.execute("""SELECT * FROM todos""")
        tasks = c.fetchall()
        conn.close()
        trigger_task_event("list_tasks", "", "Tasks retrieved successfully", "success")
    except Exception as e:
        trigger_task_event("list_tasks", "", str(e), "error")
        raise e
    return tasks


@mcp.tool(name="deletetask", description="Delete a task from the todo list by ID")
def delete_task(task: str,completed: str):
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("""DELETE FROM todos WHERE task=? and completed=?""", (task,completed))
        deleted_count = c.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count == 0:
            trigger_task_event("delete_task", task, "Task not found", "error")
            return f"Error: Task with task name {task} not found."
            
        trigger_task_event("delete_task", task, "Task deleted successfully", "success")
        return "Task deleted successfully"
    except Exception as e:
        trigger_task_event("delete_task", task, str(e), "error")
        raise e



@mcp.tool(name="updatetask", description="Update a task in the todo list by ID")
def update_task(task: str,completed: str):
    try:
        
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        if completed == "completed":
            c.execute("""UPDATE todos SET completed=? WHERE task=?""", (completed, task))
        else:
            c.execute("""UPDATE todos SET task=? WHERE task=? and completed=?""", 
                  (task,completed))
        updated_count = c.rowcount
        conn.commit()
        conn.close()

        
        if updated_count == 0:
            trigger_task_event("update_task", task, "Task not found", "error")
            return f"Error: Task with task name {task} not found."
            
        trigger_task_event("update_task", task, "Task updated successfully", "success")
        return "Task updated successfully"
    except Exception as e:
        trigger_task_event("update_task", task, str(e), "error")
        raise e



init_db()




if __name__ == "__main__":
    mcp.run()
