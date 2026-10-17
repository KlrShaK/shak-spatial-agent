# 04 — Shak's findings on AgentV5 (verbatim)

Your manual review of AgentV5 against AgentV4, shared 2026-10-17. The text below is kept exactly
as written so we can both refer back to it. Each point has an ID (T1–T16 for tools, P1–P12 for the
pipeline) in [05_discussion_tracker.md](05_discussion_tracker.md), where its discussion status is
recorded.

---

# Measuring tools & are they worth it.
I question all the available tools, family, kind and decision workflow; agentV4 had much less tool and yet performed better, hence accuracy is improved not by number of tools but efficient usage of available tools.

- Entity Track / Entities -> doesn’t output a LLM readable summary. Gives messy raw numbers. Instead of trajectory summary

- Object & projected position -> two tools with similar purpose none of them performing their job properly. Simplify. One rotation tool give complete overview of subject/subjects rotation throughout the scene or specified time interval
- Kinematics / Motion Profile → Redundant(if entity track gives motion profile) if it’s related to only speed velo/acceleration should be better named.

- Camera Profile → Name as Camera track,(motion of camera with respect to world or a subject; default - world)

- Orientation → Could be useful need to specify when to call orbit? and when to call rotation tool. (should have in tool call if the question requests. orientation at specific instance or with respect specific subj. then use this otherwise)

- Camera motion / Camera profile ->  Confusing

- Relative Position -> Need to rename relative position b/w 2 objects or
merge in entity track. (entity track could give the motion profile of multiple objects, but if relative position or motion is asked this could be a good tool)

- Elevation -> totally Redundant.

- Region Relation → Seems like entity track could answer the question

- Group Geometry → Multi object relation; Use when you need to know how multiple objects move through and distances b/w them. (to measure how some physical quantity such as rel. distance/speed, evolves between them) (Should receive all tracked entities) (should be named more accurately)

- View State/Census —> detect all entities and return the visibility and timeline. (I feel this could be available before family or kind selection, because it can also help in grounding). The timeline and view percentage can help determine false positives.

- Camera Zoom —> Keep.

- Scene revisit → How does it work? (I feel like its one of those questions where we should just ask VLM to answer- we don't have any tools for scene recognition)

- Evaluate Predicate ->  Remove

- Event Reduce / locate event -> Remove . e.g How many times did the subject raise arm", we don't have any tool to recognize event, its a completely visual question ask VLM to answer question.

- Maybe in Entity track we can give option to mention main entity & other entity. And give information relative to the main entity. Or have a different tool for relative motion (design choice, brainstorm with me.)



## What I don't like about current pipeline (Major Issues) + optional proposals

- (GENERAL FEEDBACK) A lot of tool failures.
    - I propose maybe an agentic doctor
        - An agent which have complete system design information especially about tools and work flow. 
            - Dynamically recieves more specific information about the specific tool that caused an error. For instance, the agentic doctor is called for a problem in grounding tool call, then it should receive more information about grounding tool like exactly what parameters it expects in which format, etc.
        - if the main reasoning agent outputs a misformed or erratic json tool call. 
        - The agent doctor receives the tool call along with the past history
        - The job of the doctor is 
            - 1: to understand what did the main agent wanted to do
            - 2: It determines the specific type of error in the query like if it is misformed, i.e missing some parameters or parameters are mis-spelt, etc.
            - 3: Determines if it can fix the tool call given the history and interpolate missing parameters or modify them appropriately in a format the host expects.
                - 3.b: If it cannot fix it then return specific instructions back to agent to fix the tool call.(Keep the output short, 2-3 lines)
            - 4: Return the output of the fixed tool call back to the main agent 

- (GENERAL FEEDBACK) There's often a mismatch b/w final measurement and the final answer, their reasoning doesn’t not agreeing with each other
    - Often after the measurement, the agent is giving the right reasoning that aligns with GT. But then in the final step the reasoning is incorrect and it flips towards wrong answer.

- (GENERAL FEEDBACK) The measurement tool outputs are not LLM readable and interpretable, its too complex and contains "raw data, instead of a structured summary". Refer AgentV4, it had a much better way of giving information to LLM that I would like to keep and build upon

- What the fuck is generated frame indices especially inspect.
- Improve the agent prompt - the agent is not a video analyser its a Spatial reasoning agent that answers questions about 3D motion and objects by using available tools 
- Model should ask which physical quantities do I need to measure to answer this question. 
    - It should plan do I need the distances b/w observer and object or b/w two objects or their spatial Relationship or speed, etc.
    - Then given information about its available tools decide what it should call and measure before giving answer

- Tools/Family/kind should be named better - it should be immediately be clear from the name its purpose. 
- Maybe a short description of video could be useful as well. If an egocentric video; observer=subject in this case the track observer 

- Also important to know when using tools is the timeframe, do you need to use the tool over 1 complete timeframe or tools need to be used a multiple different time-stamps 
     - possibly also good idea to seprate tools b/w those that give
overall “perspective of video”(for example complete motion profile over the selected time frame) or those that should be used multiple times at different time-stamps. The latter part could be even removed by incorporating history in tools for instance a question asks how does the distance change between two objects. instead of letting the model choose different time stamps. we could just give the relative position/motion profile over the whole video and then let the model reason, i think its a simpler/better approach.

- After a grounding failure → 1 forced retry by giving it a frame and asking a different name. append in current instructions to Not give plural names mention specific object (SAM will find multiple objects of the same class automatically, plural confuses the SAM model)
- Maybe include specific example for each tool (short sentence, that mentions when and when not to use tool) e.g Use track camera if knowing camera traj & orientation could help you answer.

- We should allow combination of tools to answer a question; eg. question "how many fishes are moving right?" -> first question is about counting -> second the timeline or Region of interest is complete video -> third detect all fishes or entities across timeline (view state/census) -> fourth track entities throught time -> sixth filter entities moving right ( in frame-of-reference camera/observer)
    - The agent can plan high-level steps before calling any tool but then it must be flexible to adapt based on outputs of the measurements recieved.
