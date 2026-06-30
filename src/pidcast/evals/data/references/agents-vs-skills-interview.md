---
title: Don't Build Agents, Build Skills Instead – Barry Zhang & Mahesh Murag, Anthropic
date: 2025-12-12
transcribed: "2025-12-12T06:49:24.339784"
url: "https://www.youtube.com/watch?v=CEvIs9y1uog"
duration: "16:22"
channel: AI Engineer
tags: ['podcast', 'youtube', 'transcription']
---

 (upbeat music)
 - All right, good morning.
 And thank you for having us again.
 Last time we were here,
 we're still figuring out what an agent even is.
 Today, many of us are using agents on a daily basis.
 But we still notice gaps.
 We still have slots, right?
 Agents have intelligence and capabilities,
 but not always expertise that we need for real work.
 And Barry, this is Mejesh, we created agent skills.
 In this talk, we'll show you why we stopped building agents
 and started building skills instead.
 A lot of things have changed since our last talk.
 MCP became the standard for agent connectivity,
 called code, our first coding agent launched to the world,
 and our cloud agent SDK now provides
 a production ready agent out of the box.
 We have a more mature ecosystem,
 and we're moving towards a new paradigm for agents.
 That paradigm is a tighter coupling
 between the model and the runtime environment.
 But simply, we think code is all we need.
 We used to think agents in different domains
 will look very different.
 Each one, one need its own tools and scaffolding,
 and that means we'll have a separate agent
 for each use case for each domain.
 Well, customization is still important for each domain.
 The agent underneath is actually more universal
 than we thought.
 What we realize is that code is not just a use case,
 but the universal interface to the digital world.
 After we built call code,
 we realized that call code is actually a general purpose agent.
 Think about generating a financial report.
 The model can call the API to point data and do research.
 You can organize that data in the file system,
 you can analyze it with Python,
 and then synthesize the insight in all file format,
 all through code.
 The core scaffolding can suddenly become as thin
 as just bash and file system,
 which is great and really scalable,
 but we very quickly run into a different problem.
 And that problem is domain expertise.
 Who do you want doing your taxes?
 Is it gonna be Mejesh, the 300 IQ mathematical genius,
 or is it Barry, an experienced tax professional?
 I would pick Barry every time.
 I don't want Mejesh to figure out the 2025 tax code
 from first principles,
 and he consistent execution from a domain expert.
 Agents today are a lot like Mejesh,
 they're brilliant, but they lack expertise.
 (audience laughs)
 They can do, no more slow.
 They can do amazing things
 when you really put an effort and give proper guidance,
 but they're often missing the important context upfront,
 they can't really absorb your expertise super well,
 and they don't learn over time.
 That's why we created Agent Skills.
 Skills are organized collections of files
 that package composable procedural knowledge for agents.
 In other words, they're folders.
 This simplicity is deliberate.
 We want something that anyone, human or agent,
 can create and use as long as they have a computer.
 You've also worked with what you already have.
 You can version them in Git,
 you can throw them in Google Drive,
 and you can zip them up and share it with your team.
 We have used files as a primitive for decades,
 and we like them, so why change now?
 Because of that, skills can also include
 a lot of scripts as tools.
 Traditional tools have pretty obvious problems.
 Some tools have poorly written instructions
 that are pretty ambiguous,
 and when the model is struggling,
 it can really make a change to the tool,
 so it's just kind of stuck with a code start problem,
 and they always live in the context window.
 Code solves some of these issues.
 It's self-documenting, it is modifiable,
 and it can live in the file system
 until they're really needed and used.
 Here's an example of a script inside of a skill.
 We kept seeing Cloud write the same Python script
 over and over again to apply styling to slides,
 so we just asked Cloud to save it inside of the skill
 as a tool for its future self.
 Now we can just run the script,
 and that makes everything a lot more consistent,
 a lot more efficient.
 At this point, skills can contain a lot of information,
 and we want to protect the context window
 so that we can fit in hundreds of skills
 and make them truly composable.
 That's why skills are progressively disclosed.
 At runtime, only this metadata is shown to the model
 just to indicate that it has the skill.
 When an agent needs to use a skill,
 you can read in the rest of the skill.md,
 which contains the core instruction and directory
 for the rest of the folder.
 Everything else is just organized for ease of access.
 So that's all skills are,
 they're organized folders with scripts as tools.
 - Since our launch five weeks ago,
 this very simple design has translated
 into a very quickly growing ecosystem
 of thousands of skills.
 And we've seen this be split across
 a couple of different types of skills.
 There are foundational skills, third-party skills
 created by partners in the ecosystem,
 and skills built within an enterprise and within teams.
 To start, foundational skills are those that give agents
 new general capabilities or domain-specific capabilities
 that it didn't have before.
 We ourselves with our launch built document skills
 that give Claude the ability to create
 and edit professional quality office documents.
 We're also really excited to see people like Cadence
 build scientific research skills
 that give Claude new capabilities like EHR data analysis
 and using common Python bioinformatics libraries
 better than it could before.
 We've also seen partners in the ecosystem build skills
 that help Claude better with their own software
 and their own products.
 Browser-based is a pretty good example of this.
 They built a skill for their open-source browser
 automation tooling stage-hand.
 And now Claude equipped this skill and with stage-hand
 can now go navigate the web and use a browser
 more effectively to get work done.
 And Notion launched a bunch of skills
 that help Claude better understand your Notion workspace
 and do deep research over your entire workspace.
 And I think where I've seen the most excitement
 and traction with skills is within large enterprises.
 These are company and team-specific skills
 built for an organization.
 We've been talking to Fortune 100s
 that are using skills as a way to teach agents
 about their organizational best practices
 and the weird and unique ways that they use
 this bespoke internal software.
 We're also talking to really large developer productivity teams.
 These are teams serving thousands or even tens of thousands
 of developers in an organization that are using skills
 as a way to deploy agents like Claude Code
 and teach them about code style best practices
 and other ways that they want their developers
 to work internally.
 So all of these different types of skills are created
 and consumed by different people inside of an organization
 or in the world, but what they have in common
 is anyone can create them and they give agents
 the new capabilities that they didn't have before.
 So as this ecosystem has grown,
 we've started to observe a couple of interesting trends.
 First, skills are starting to get more complex.
 The most basic skill today can still be
 a skill.md markdown file with some prompts
 and some really basic instructions.
 But we're starting to see skills that package software,
 executables, binaries, files, code scripts, assets
 and a lot more.
 And a lot of the skills that are being built today
 might take minutes or hours to build
 and put into an agent.
 But we think that increasingly much like
 a lot of the software we use today,
 these skills might take weeks or months to build
 and be maintained.
 We're also seeing that this ecosystem of skills
 is complementing the existing ecosystem of MTP servers
 that was built up over the course of this year.
 Developers are using and building skills
 that orchestrate workflows of multiple MCP tools
 stitched together to do more complex things
 with external data and connectivity.
 And in these cases, MCP is providing the connection
 to the outside world, while skills are providing
 the expertise.
 And finally, and I think most excitingly for me personally,
 is we're seeing skills that are being built
 by people that aren't technical.
 These are people in functions like finance, recruiting,
 accounting, legal, and a lot more.
 And I think this is pretty early validation
 of our initial idea that skills help people that
 aren't doing coding work extend these general agents.
 And they make these agents more accessible for the day
 to day of what these people are working on.
 So tying this all together, let's talk about how these all fit
 into this emerging architecture of general agents.
 First, we think this architecture
 is converging on a couple of things.
 The first is this agent loop that
 helps manage the model's internal context
 and manages what tokens are going in and out.
 And this is coupled with a runtime environment
 that provides the agent with a file system
 and the ability to read and write code.
 This agent, as many of us have done
 throughout this year, can be connected to MCP servers.
 And these are tools and data from the outside world
 that make the agent more relevant and more effective.
 And now we can give the same agent a library
 of hundreds or thousands of skills
 that it can decide to pull into context only at runtime
 when it's deciding to work on a particular task.
 Today, giving an agent a new capability in a new domain
 might dis-involve equipping it with the right set of MCP
 servers and the right library of skills.
 And this emerging pattern of an agent with an MCP server
 and a set of skills is something that's already helping us
 at anthropic deploy Claude to new verticals.
 Just after we launched skills five weeks ago,
 we immediately launched new offerings in financial services
 and life sciences.
 And each of these came with a set of MCP servers
 and a set of skills that immediately make Claude more
 effective for professionals in each of these domains.
 We're also starting to think about some of the other open
 questions and areas that we want to focus on for how
 skills evolve in the future.
 As they start to become more complex,
 we really want to support developers, enterprises,
 and other skill builders by starting
 to treat skills like we treat software.
 This means exploring testing and evaluation.
 Better tooling to make sure that these agents
 are loading and triggering skills at the right time
 and for the right task.
 And tooling to help measure the output quality of an agent
 equipped with a skill to make sure that's on par
 with what the agent is supposed to be doing.
 We'd also like to focus on versioning.
 As a skill evolves and the resulting agent behavior evolves,
 we want this to be clearly tracked and to have a clear
 lineage over time.
 And finally, we'd also like to explore skills that can
 explicitly depend on and refer to either other skills,
 MCP servers, and dependencies and packages
 within the agent's environment.
 We think that this is going to make agents a lot more
 predictable in different runtime environments.
 And the composability of multiple skills together
 will help agents like Claude elicit even more complex
 and relevant behavior from these agents.
 Overall, these set of things should hopefully make skills
 easier to build and easier to integrate
 into agent products, even those besides Claude.
 Finally, a huge part of the value of skills
 we think is going to come from sharing and distribution.
 Barry and I think a lot about the future of companies
 that are deploying these agents at scale.
 And the vision that excites us most
 is one of a collecting and a collective and evolving
 knowledge base of capabilities that's
 curated by people and agents inside of an organization.
 We think skills are a big step towards this vision.
 They provide the procedural knowledge
 for your agents to do useful things.
 And as you interact with an agent and give it feedback
 and more institutional knowledge,
 it starts to get better.
 And all of the agents inside your team
 and your org get better as well.
 And when someone joins your team
 and starts using Claude for the first time,
 it already knows what your team cares about.
 It knows about your day-to-day.
 And it knows about how to be most effective for the work
 that you're doing.
 And as this grows and this ecosystem starts to develop even
 more, this compound value is going
 to extend outside of just your org
 and into the broader community.
 So just like when someone else across the world builds
 an MCP server that makes your agent more useful,
 a skill built by someone else in the community
 will help make your own agents more capable, reliable,
 and useful as well.
 This vision of an evolving knowledge base
 gets even more powerful when Claude starts
 to create these skills.
 We design skills specifically as a concrete step
 towards continuous learning.
 When you first start using Claude,
 this standardized format gives a very important guarantee.
 Anything that Claude writes down
 can be used efficiently by a future version of itself.
 This makes the learning actually transferable.
 As you build up the context,
 skills makes the concept of memory more tangible.
 They don't capture everything.
 They don't capture every type of information.
 Just procedural knowledge that Claude can use
 on specific tasks.
 We have worked with Claude for quite a while.
 The flexibility of skills matters even more.
 Claude can acquire new capabilities instantly,
 evolve them as needed,
 and then drop the ones that become obsolete.
 This is what we have always known.
 The power in context learning makes this
 a lot more cost effective for information
 that change on a daily basis.
 Our goal is that Claude on day 30 of working with you
 is going to be a lot better on Claude on day one.
 Claude can already create skills for you today
 using our skill creator skill,
 and we're going to continue pushing in that direction.
 We're going to conclude by comparing the agent stack
 to what we have already seen in computing.
 In a rough analogy, models are like processors.
 Both require massive investment
 and contain immense potential.
 They're only so useful by themselves.
 Then we start building operating system.
 The OS made processors far more valuable
 by orchestrating the processes, resources, and data
 around the processor.
 In AI, we believe the agent runtime
 is starting to play this role.
 We're all trying to build the cleanest,
 most efficient, and most scalable abstractions
 to get the right tokens in and out of the model.
 But once we have a platform,
 the real value comes from applications.
 A few companies build processors and operating systems,
 but millions of developers, like us,
 have built softwares that encoded domain expertise
 and are unique points of view.
 We hope that skills can help us open up this layer
 for everyone.
 This is where we get creative
 and solve concrete problems for ourselves,
 for each other and for the world,
 just by putting stuff in the folder.
 So skills are just the starting point.
 - To close out, we think we're now converging
 on this general architecture for general agents.
 We've created skills as a new paradigm
 for shipping and sharing new capabilities.
 So we think it's time to stop rebuilding agents
 and start building skills instead.
 And if you're excited about this,
 come work with us and start building some skills today.
 Thank you.
 (applause)
 (upbeat music)
 (upbeat music)
