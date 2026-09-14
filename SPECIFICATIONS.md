# Private RAG That Runs on Local Compute and Uses Gemma model hosted on Google AI Studio to Generate the Replies
## Requirements
1. Build a web app with three pages switchable by a tab or button at the top of the window.
2. The first page takes a url or local directory, then create a private vector database from documents in the url or local directory and save it to database/ folder. Use local ollama to create the vector database.
3. The second page takes a question from the user and generate an answer based on the documents in the vector database using Gemma model hosted on Google AI Studio.
4. All of the backend code should be written in python, including local ollama, the web app should be built using flask.
5. All calls to external resources like Google AI Studio or any external API should be done from the backend python code, not from the frontend.
6. All the calls to ollama or external resources should be logged in database/logs.json. The log should include the time of the call, the type of the call, the arguments passed to the call, and the response from the call. Exclude the vector or non-text content in the call.
7. Add agent skills capability following the SKILL_SPEC.md file. Automatically discover, embed and index skills when the web app starts. Use local ollama to create the vector database for skills.Create the following skills:
   - get the time and weather of the city from the site that doesn't require API key
   - get the list of stocks with the highest percentage increase or lowest percentage decrease based on the chat question
8. Create a README.md with a brief description about:
   - what this system does
   - how to install the components needed
   - how to start all the services
   - how to shutdown all the services
   - user guide with information on how to use the system