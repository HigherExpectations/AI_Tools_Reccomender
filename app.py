# ============================================================
# AI Tool Recommender - FIXED & IMPROVED (Part 1: Data)
# ============================================================
import os, re, json, math, random
from datetime import datetime, date
from urllib.parse import urlparse
from collections import Counter
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

TODAY = date.today().isoformat()

# ---------------------------------------------------------------
# 1. DATABASE
# ---------------------------------------------------------------
DATASET_JSON = r'''[
{"name":"ChatGPT","category":"General AI","description":"general purpose ai assistant writing coding reasoning research chat","url":"https://chatgpt.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free tier; Plus $20/mo"},
{"name":"Claude","category":"General AI","description":"ai assistant long document analysis coding reasoning writing","url":"https://claude.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free tier; Pro $20/mo"},
{"name":"Gemini","category":"General AI","description":"google multimodal ai search research writing coding workspace","url":"https://gemini.google.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Advanced $19.99/mo"},
{"name":"Perplexity","category":"General AI","description":"ai search engine web search answer questions citations","url":"https://perplexity.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Pro $20/mo"},
{"name":"DeepSeek","category":"General AI","description":"reasoning math logic coding general assistant open source","url":"https://deepseek.com","pricing_model":"Free","rating":"4.8/5","pricing_details":"Free web; API pay-per-token"},
{"name":"Microsoft Copilot","category":"General AI","description":"microsoft assistant web search text image generation bing","url":"https://copilot.microsoft.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Copilot Pro $20/mo"},
{"name":"Pi","category":"General AI","description":"personal companion empathetic conversation advice support","url":"https://pi.ai","pricing_model":"Free","rating":"4.5/5","pricing_details":"Completely free"},
{"name":"Character.ai","category":"General AI","description":"roleplay character chat fictional celebrity persona entertainment","url":"https://character.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; c.ai+ $9.99/mo"},
{"name":"Poe","category":"General AI","description":"multiple chatbots aggregator platform custom bots access","url":"https://poe.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free daily points; $19.99/mo"},
{"name":"You.com","category":"General AI","description":"ai search engine research web assistant multiple models","url":"https://you.com","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free; YouPro $20/mo"},
{"name":"Jasper","category":"Copywriting & Marketing","description":"marketing copywriting brand voice ad copy business content","url":"https://jasper.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"$39/creator/mo; 7-day trial"},
{"name":"Copy.ai","category":"Copywriting & Marketing","description":"marketing copywriting blog post social media content automation","url":"https://copy.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free tier; Pro $49/mo"},
{"name":"Writesonic","category":"Copywriting & Marketing","description":"seo article blog writing content optimization marketing copy","url":"https://writesonic.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free trial; ~$15-20/mo"},
{"name":"Rytr","category":"Copywriting & Marketing","description":"short form content writing emails posts social media quick copy","url":"https://rytr.me","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free 10k chars; $9/mo unlimited"},
{"name":"Anyword","category":"Copywriting & Marketing","description":"predictive performance scoring ad copy marketing copywriting","url":"https://anyword.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"7-day trial; $39/mo"},
{"name":"Sudowrite","category":"Creative & Fiction Writing","description":"story novel writing creative fiction plot character dialogue","url":"https://sudowrite.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Trial; $10/mo"},
{"name":"NovelAI","category":"Creative & Fiction Writing","description":"story writing anime fiction literature narrative creative","url":"https://novelai.net","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Trial; $10/mo"},
{"name":"Squibler","category":"Creative & Fiction Writing","description":"screenwriting script novel book writing story outline","url":"https://squibler.io","pricing_model":"Freemium","rating":"4.2/5","pricing_details":"Free; Pro $16/mo"},
{"name":"QuillBot","category":"Paraphrasing & Grammar","description":"paraphrase rewrite grammar check summarize rephrase text","url":"https://quillbot.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free limits; $9.95/mo"},
{"name":"Grammarly","category":"Paraphrasing & Grammar","description":"grammar check spell check writing assistant proofread tone","url":"https://grammarly.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; $12/mo"},
{"name":"Wordtune","category":"Paraphrasing & Grammar","description":"rewrite sentence rephrase text tone adjust writing improvement","url":"https://wordtune.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"10 rewrites/day; $9.99/mo"},
{"name":"ProWritingAid","category":"Paraphrasing & Grammar","description":"writing editor grammar style check critique analysis readability","url":"https://prowritingaid.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free limits; $10/mo"},
{"name":"LanguageTool","category":"Paraphrasing & Grammar","description":"grammar spell check multilingual proofread correction","url":"https://languagetool.org","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; $4.99/mo"},
{"name":"ChatPDF","category":"PDF & Document Chat","description":"pdf chat document question answer file upload analyze extract","url":"https://chatpdf.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"2 PDFs/day; $19.99/mo"},
{"name":"Humata","category":"PDF & Document Chat","description":"pdf document chat summarize extract data research papers","url":"https://humata.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"60 pages/mo; $1.99/mo student"},
{"name":"Scite","category":"Research & Citations","description":"smart citations research paper scientific literature validate","url":"https://scite.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"7-day trial; ~$12/mo"},
{"name":"Article Forge","category":"Content Generation","description":"article generation seo content bulk writing automation","url":"https://articleforge.com","pricing_model":"Paid","rating":"4.1/5","pricing_details":"5-day trial; $27/mo"},
{"name":"WordAI","category":"Content Generation","description":"article rewriter spinner paraphrase content bulk rewrite","url":"https://wordai.com","pricing_model":"Paid","rating":"4.2/5","pricing_details":"3-day trial; $27/mo"},
{"name":"Ink for All","category":"SEO & Content","description":"seo content writing optimization marketing editor search","url":"https://inkforall.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Trial; $39/mo"},
{"name":"DeepL Write","category":"Paraphrasing & Grammar","description":"writing improvement grammar rephrase multilingual translation style","url":"https://deepl.com/write","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Pro $8.74/mo"},
{"name":"Midjourney","category":"Image Generation","description":"image picture art draw logo photo illustration realistic artistic","url":"https://midjourney.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$10/mo Basic"},
{"name":"DALL-E 3","category":"Image Generation","description":"image generation picture art draw text to image openai","url":"https://openai.com/dall-e-3","pricing_model":"Paid","rating":"4.7/5","pricing_details":"ChatGPT Plus $20/mo or API"},
{"name":"Stable Diffusion","category":"Image Generation","description":"open source image generation art photo model stable diffusion","url":"https://stability.ai","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free open source; API pay-as-you-go"},
{"name":"Leonardo AI","category":"Image Generation","description":"game assets image art character design illustration creative","url":"https://leonardo.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"150 daily tokens; $10/mo"},
{"name":"Ideogram","category":"Image Generation","description":"text in image typography logo design lettering signage","url":"https://ideogram.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free slow credits; $7/mo"},
{"name":"Adobe Firefly","category":"Image Generation","description":"photoshop image edit generate art commercial safe design","url":"https://firefly.adobe.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free credits; $4.99/mo"},
{"name":"Canva AI","category":"Graphic Design","description":"graphic design template image presentation social media poster","url":"https://canva.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Pro $15/mo"},
{"name":"Krea AI","category":"Image Generation","description":"real-time image generation enhance upscale visual paint","url":"https://krea.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free limits; Pro $24/mo"},
{"name":"NightCafe","category":"Image Generation","description":"art style image creation painting community gallery","url":"https://creator.nightcafe.studio","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free daily credits; $4.79/mo"},
{"name":"Magnific AI","category":"Image Upscaling","description":"upscale enhance image resolution detail add sharpen","url":"https://magnific.ai","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$39/mo"},
{"name":"Topaz Photo AI","category":"Image Upscaling","description":"denoise sharpen upscale photo enhance quality noise reduction","url":"https://topazlabs.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$199 one-time"},
{"name":"Remove.bg","category":"Background Removal","description":"remove background image transparent cutout erase photo","url":"https://remove.bg","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free low-res; credits for HD"},
{"name":"Upscayl","category":"Image Upscaling","description":"free image upscale enlarge resolution open source enhance local","url":"https://upscayl.org","pricing_model":"Free","rating":"4.7/5","pricing_details":"100% free open source"},
{"name":"Photoroom","category":"Photo Editing","description":"product photo edit background remove studio ecommerce","url":"https://photoroom.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Pro $12.99/mo"},
{"name":"Clipdrop","category":"Photo Editing","description":"relight cleanup remove text object image photo edit","url":"https://clipdrop.co","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; ~$7-10/mo"},
{"name":"Let's Enhance","category":"Image Upscaling","description":"upscale image quality improve resolution color enhance","url":"https://letsenhance.io","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"10 credits; $9/mo"},
{"name":"Artbreeder","category":"Image Generation","description":"image breeding morph mix portrait landscape character blend","url":"https://artbreeder.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free credits; $8.99/mo"},
{"name":"DreamStudio","category":"Image Generation","description":"stable diffusion image generation art photo text to image","url":"https://dreamstudio.ai","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Credit-based; $10=1000 credits"},
{"name":"Playground AI","category":"Image Generation","description":"image generation art design creative edit free canvas","url":"https://playground.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free daily; Pro $15/mo"},
{"name":"Recraft","category":"Vector & Design","description":"vector image svg illustration icon design brand","url":"https://recraft.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free public; Pro $20/mo"},
{"name":"Lensa AI","category":"Photo Editing","description":"photo editing avatar portrait selfie magic correction","url":"https://lensa.app","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free trial; ~$29.99/yr"},
{"name":"Fotor","category":"Photo Editing","description":"photo editor image design collage retouch enhance","url":"https://fotor.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Pro $3.33/mo"},
{"name":"Picsart","category":"Photo Editing","description":"photo editor collage maker design social media image","url":"https://picsart.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Plus $5/mo"},
{"name":"Luminar Neo","category":"Photo Editing","description":"photo editing ai enhance sky replace portrait retouch lighting","url":"https://skylum.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"~$11.95/mo or lifetime"},
{"name":"HeadshotPro","category":"Headshot Generation","description":"professional headshot portrait corporate linkedin photo","url":"https://headshotpro.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"$29 for 40 headshots"},
{"name":"Aragon AI","category":"Headshot Generation","description":"professional headshot portrait avatar linkedin business","url":"https://aragon.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"$35 for 20 headshots"},
{"name":"Stockimg AI","category":"Stock Image Generation","description":"stock image generation book cover poster wallpaper design","url":"https://stockimg.ai","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free limits; $19/mo"},
{"name":"Booth AI","category":"Product Photography","description":"generative ai platform e-commerce lifestyle product photography","url":"https://booth.ai","pricing_model":"Paid","rating":"4.2/5","pricing_details":"Trial; $199/mo"},
{"name":"VistaCreate","category":"Graphic Design","description":"graphic design template social media poster flyer","url":"https://create.vista.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Pro $10/mo"},
{"name":"Visme AI","category":"Infographic & Visual","description":"infographic presentation visual design chart diagram","url":"https://visme.co","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Starter $12.25/mo"},
{"name":"Runway Gen-2","category":"Video Generation","description":"video movie clip film animate generation text to video edit","url":"https://runwayml.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"125 credits; Standard $12/mo"},
{"name":"Pika Labs","category":"Video Generation","description":"video generation animation text to video 3d motion","url":"https://pika.art","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free credits; Standard $8/mo"},
{"name":"Sora","category":"Video Generation","description":"openai video generation realistic film minute long","url":"https://openai.com/sora","pricing_model":"Paid","rating":"4.8/5","pricing_details":"ChatGPT Plus/Pro tiers"},
{"name":"Synthesia","category":"AI Video Avatars","description":"avatar video presentation corporate talking head training","url":"https://synthesia.io","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Starter $18/mo"},
{"name":"HeyGen","category":"AI Video Avatars","description":"avatar video translate lip sync talking presenter multilingual","url":"https://heygen.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"1 credit; Creator $24/mo"},
{"name":"Invideo AI","category":"Video Generation","description":"video script youtube content creation auto edit generate","url":"https://invideo.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free watermarks; Plus $20/mo"},
{"name":"Fliki","category":"Video Generation","description":"text to video voiceover blog to video social media content","url":"https://fliki.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"5 min/mo; Standard $21/mo"},
{"name":"Luma Dream Machine","category":"Video Generation","description":"video generation 3d render animate realistic high quality","url":"https://lumalabs.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free monthly; Standard $23.99/mo"},
{"name":"Kaiber","category":"Video Generation","description":"music video audio reactive animation transform visual effects","url":"https://kaiber.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"7-day trial; $5/mo"},
{"name":"Genmo","category":"Video Generation","description":"video generation motion 3d animate interactive creative","url":"https://genmo.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free daily; Turbo $10/mo"},
{"name":"Descript","category":"Video & Podcast Editing","description":"video editing podcast audio transcript text based edit","url":"https://descript.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Creator $12/mo"},
{"name":"Opus Clip","category":"Video Clipping","description":"short clip video edit viral tiktok youtube shorts repurpose","url":"https://opus.pro","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free minutes; Starter $9/mo"},
{"name":"Veed.io","category":"Video Editing","description":"video editor subtitle transcription online screen record","url":"https://veed.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free watermark; Lite $12/mo"},
{"name":"Pictory","category":"Video Generation","description":"text to video article to video script video summary","url":"https://pictory.ai","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Trial; Standard $19/mo"},
{"name":"Steve AI","category":"Video Generation","description":"animation video cartoon explainer marketing create","url":"https://steve.ai","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free watermark; Basic $15/mo"},
{"name":"Elai.io","category":"AI Video Avatars","description":"avatar video presentation corporate training text to video","url":"https://elai.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"1 min trial; Basic $23/mo"},
{"name":"Colossyan","category":"AI Video Avatars","description":"avatar video workplace training corporate presentation","url":"https://colossyan.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Trial; Starter $19/mo"},
{"name":"DeepBrain","category":"AI Video Avatars","description":"ai avatar video human realistic presentation news","url":"https://deepbrain.io","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Starter $24/mo"},
{"name":"D-ID","category":"AI Video Avatars","description":"talking head avatar photo to video face animation","url":"https://d-id.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"14-day trial; Lite $5.90/mo"},
{"name":"Wisecut","category":"Video Editing","description":"video editing auto cut silence remove smart edit","url":"https://wisecut.video","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"45 min/mo; Starter $10/mo"},
{"name":"Munch","category":"Video Clipping","description":"video repurpose clip social media tiktok shorts extract","url":"https://getmunch.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$49/mo for 200 min"},
{"name":"Topview AI","category":"Video Generation","description":"video ad marketing social media create online ecommerce","url":"https://topview.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free credits; ~$29/mo"},
{"name":"Gling","category":"Video Editing","description":"video editing youtube creator cut silence trim filler","url":"https://gling.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"First video free; $15/mo"},
{"name":"Suno AI","category":"Audio Generation","description":"music song audio track beat singing vocals melody generation","url":"https://suno.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"50 credits/day; Pro $10/mo"},
{"name":"Udio","category":"Audio Generation","description":"music generation song audio high quality vocals lyrics","url":"https://udio.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free monthly; Standard $10/mo"},
{"name":"ElevenLabs","category":"Voice & TTS","description":"voice voiceover speech narration talking dubbing tts cloning","url":"https://elevenlabs.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"10k chars/mo; Starter $5/mo"},
{"name":"Murf AI","category":"Voice & TTS","description":"voiceover presentation professional voice text to speech tts","url":"https://murf.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"10 min free; Creator $23/mo"},
{"name":"Speechify","category":"Voice & TTS","description":"text to speech read aloud audiobook dyslexia speed listening","url":"https://speechify.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free voices; ~$139/yr"},
{"name":"AIVA","category":"Audio Generation","description":"soundtrack background music film score compose orchestral","url":"https://aiva.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free non-commercial; €11/mo"},
{"name":"Soundraw","category":"Audio Generation","description":"royalty free music background beat generate license","url":"https://soundraw.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free preview; $16.99/mo"},
{"name":"Boomy","category":"Audio Generation","description":"music creation beat lofi generate upload share produce","url":"https://boomy.com","pricing_model":"Freemium","rating":"4.2/5","pricing_details":"Free create; Creator $9.99/mo"},
{"name":"Beatoven","category":"Audio Generation","description":"background music video podcast royalty free stream license","url":"https://beatoven.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free limits; Pro $6/mo"},
{"name":"Lalal.ai","category":"Audio Splitting","description":"stem splitter vocal remove isolate track extract instrument","url":"https://lalal.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"10 min trial; $15 pack"},
{"name":"Play.ht","category":"Voice & TTS","description":"text to speech voice generation audio content podcast ultra realistic","url":"https://play.ht","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"12.5k chars; Creator $31.20/mo"},
{"name":"Resemble AI","category":"Voice Cloning","description":"voice cloning custom tts speech synthesis dubbing deepfake","url":"https://resemble.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"$0.006/sec; $29/mo"},
{"name":"WellSaid Labs","category":"Voice & TTS","description":"voiceover narration professional corporate training tts enterprise","url":"https://wellsaidlabs.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"7-day trial; Maker $44/mo"},
{"name":"LOVO AI","category":"Voice & TTS","description":"voice generation tts voiceover video narration multilingual 100 languages","url":"https://lovo.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"14-day trial; Basic $24/mo"},
{"name":"Kits AI","category":"Voice Cloning","description":"voice model singing vocal clone music artist voice model","url":"https://kits.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; Converter $9.99/mo"},
{"name":"Musicfy","category":"Voice Cloning","description":"music voice clone singing vocal song create transform","url":"https://musicfy.lol","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free trial; ~$9/mo"},
{"name":"Moises","category":"Audio Splitting","description":"audio track separator musician practice stem split pitch","url":"https://moises.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"5 splits/mo; Premium $3.99/mo"},
{"name":"Landr","category":"Music Mastering","description":"music mastering distribution release audio production","url":"https://landr.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free preview; Studio $11.99/mo"},
{"name":"Soundful","category":"Audio Generation","description":"music generation background track royalty free create","url":"https://soundful.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"3 downloads; $59.99/yr"},
{"name":"Cleanvoice","category":"Audio Editing","description":"audio editing remove filler words um ah noise podcast cleanup","url":"https://cleanvoice.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"30 min trial; €10 credits"},
{"name":"Adobe Podcast","category":"Audio Enhancement","description":"audio enhance voice clean noise remove recording studio","url":"https://podcast.adobe.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free basic; Express $9.99/mo"},
{"name":"Altered Studio","category":"Voice Editing","description":"voice editing morph change pitch tts professional morphing","url":"https://altered.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free basic; Real-Time $6/mo"},
{"name":"GitHub Copilot","category":"AI Code Assistants","description":"code autocomplete completions inline chat ide programming","url":"https://github.com/features/copilot","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$10/mo or $100/yr"},
{"name":"Cursor","category":"AI Code Editors","description":"ai code editor ide refactoring multi-file code generation","url":"https://cursor.com","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"2000 completions; Pro $20/mo"},
{"name":"Tabnine","category":"AI Code Assistants","description":"code completion developer autocomplete privacy enterprise","url":"https://tabnine.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; $12/user/mo"},
{"name":"Codeium","category":"AI Code Assistants","description":"free copilot code autocomplete chat python javascript","url":"https://codeium.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free individuals; $12/user/mo"},
{"name":"Replit AI","category":"Cloud IDE & Agents","description":"cloud ide coding online programming agent deploy","url":"https://replit.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Core $120/yr"},
{"name":"Amazon Q","category":"AI Code Assistants","description":"aws cloud code developer assistant business support","url":"https://aws.amazon.com/q","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free tier; Pro $19/user/mo"},
{"name":"Blackbox AI","category":"AI Code Assistants","description":"code search snippet developer programming copy extract","url":"https://useblackbox.io","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free basic; $0.99/week"},
{"name":"v0 by Vercel","category":"UI & Frontend Generation","description":"generative ui react components tailwind css frontend web design","url":"https://v0.dev","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free credits; Premium $20/mo"},
{"name":"Uizard","category":"UI & Frontend Generation","description":"wireframe ui app design mockup sketch prototype screenshot to code","url":"https://uizard.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Pro $12/mo"},
{"name":"Galileo AI","category":"UI & Frontend Generation","description":"ui design figma interface mockup describe generate wireframe","url":"https://usegalileo.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Trial credits; $19/mo"},
{"name":"CodiumAI","category":"Code Testing & Review","description":"code test generation unit test coverage developer qa edge cases","url":"https://codium.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free individuals; Teams $19/user/mo"},
{"name":"Sourcery","category":"Code Testing & Review","description":"code refactor python javascript quality review clean code","url":"https://sourcery.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free OSS; Pro $10/mo"},
{"name":"CodeGeeX","category":"AI Code Assistants","description":"code generation completion multilingual programming assistant","url":"https://codegeex.ai","pricing_model":"Free","rating":"4.4/5","pricing_details":"Free for individuals"},
{"name":"AskCodi","category":"AI Code Assistants","description":"code generator helper explain documentation sql query","url":"https://askcodi.com","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"50 credits/mo; $9.99/mo"},
{"name":"Mintlify","category":"Code Documentation","description":"code documentation auto generate docs readme api","url":"https://mintlify.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Startup $150/mo"},
{"name":"Sourcegraph Cody","category":"Code Search & Chat","description":"code search codebase understanding large repository navigate chat","url":"https://sourcegraph.com/cody","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Pro $9/user/mo"},
{"name":"Aider","category":"AI Code Assistants","description":"command line cli git pair programming terminal code edit","url":"https://aider.chat","pricing_model":"Free","rating":"4.8/5","pricing_details":"Free OSS (bring your own API key)"},
{"name":"Continue.dev","category":"AI Code Assistants","description":"open source ide extension autocomplete custom llm vscode","url":"https://continue.dev","pricing_model":"Free","rating":"4.7/5","pricing_details":"100% free open source"},
{"name":"Dify","category":"AI App Building","description":"llm application platform ai app build workflow rag agent","url":"https://dify.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free sandbox; Cloud Pro $59/mo"},
{"name":"Devv AI","category":"Developer Search","description":"developer search engine programming questions documentation","url":"https://devv.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Pro $9.90/mo"},
{"name":"Snyk AI","category":"Code Security","description":"code security vulnerability open source dependency scan developer","url":"https://snyk.io","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free developers; Teams $25/dev/mo"},
{"name":"CodeGPT","category":"AI Code Assistants","description":"ide extension chat code generation error troubleshooting","url":"https://codegpt.co","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; Pro $9.99/mo"},
{"name":"Notion AI","category":"Notes & Knowledge","description":"notes productivity organize plan calendar wiki knowledge base workspace","url":"https://notion.so","pricing_model":"Paid Add-on","rating":"4.7/5","pricing_details":"$8/member/mo add-on"},
{"name":"Gamma AI","category":"Presentation Generation","description":"presentation slides powerpoint pitch deck visual document generate","url":"https://gamma.app","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"400 credits; Plus $8/mo"},
{"name":"Tome","category":"Presentation Generation","description":"presentation storytelling slides visual outline narrative pitch","url":"https://tome.app","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free credits; Pro $16/mo"},
{"name":"Beautiful.ai","category":"Presentation Generation","description":"presentation design slides powerpoint template smart layout","url":"https://beautiful.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"14-day trial; Pro $12/mo"},
{"name":"Slidesgo AI","category":"Presentation Generation","description":"presentation generator slides template design education google slides","url":"https://slidesgo.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free limits; ~$4.99/mo"},
{"name":"Mem AI","category":"Notes & Knowledge","description":"notes knowledge base organize calendar thoughts auto self-organizing","url":"https://mem.ai","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free; Teams $14.99/user/mo"},
{"name":"Taskade","category":"Project Management","description":"project management tasks agent workflow mind map outline","url":"https://taskade.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"1000 credits; Pro $8/user/mo"},
{"name":"Motion","category":"Scheduling & Calendar","description":"schedule calendar task time management auto plan priority","url":"https://usemotion.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"7-day trial; $19/mo"},
{"name":"Reclaim AI","category":"Scheduling & Calendar","description":"calendar schedule time management meeting habit block google","url":"https://reclaim.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Starter $8/user/mo"},
{"name":"Superhuman","category":"Email & Communication","description":"email inbox fast productivity ai sort reply triage follow-up","url":"https://superhuman.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$25/mo"},
{"name":"ClickUp AI","category":"Project Management","description":"project management task productivity team workflow summarize","url":"https://clickup.com","pricing_model":"Paid Add-on","rating":"4.6/5","pricing_details":"$5/user/mo add-on"},
{"name":"Asana AI","category":"Project Management","description":"project management task team workflow status update risk","url":"https://asana.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"AI in Starter $10.99/user/mo+"},
{"name":"Coda AI","category":"Notes & Knowledge","description":"document doc collaboration table automation workflow generate","url":"https://coda.io","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free credits; AI add-on"},
{"name":"Airtable AI","category":"Data & Spreadsheets","description":"database spreadsheet automation workflow table organize categorize","url":"https://airtable.com","pricing_model":"Paid Add-on","rating":"4.6/5","pricing_details":"$6/user/mo add-on"},
{"name":"Slite AI","category":"Notes & Knowledge","description":"team knowledge base notes documentation collaborate wiki","url":"https://slite.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"50 docs; Standard $8/user/mo"},
{"name":"Guru","category":"Knowledge Management","description":"company knowledge wiki search intranet information enterprise","url":"https://getguru.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; AI $15/user/mo"},
{"name":"Nuclino","category":"Notes & Knowledge","description":"team wiki knowledge base documentation collaborate notes lightweight","url":"https://nuclino.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Standard $5/user/mo"},
{"name":"Tana","category":"Notes & Knowledge","description":"notes knowledge management outline organize supertag structure","url":"https://tana.inc","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Core ~$14/mo"},
{"name":"Sunsama","category":"Scheduling & Calendar","description":"daily planner task calendar schedule time block focus consolidate","url":"https://sunsama.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"14-day trial; $16/mo"},
{"name":"Akiflow","category":"Scheduling & Calendar","description":"task scheduler inbox calendar consolidate productivity time-block","url":"https://akiflow.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"7-day trial; $19/mo"},
{"name":"Consensus","category":"Research & Academia","description":"scientific research paper search study evidence based answer academic","url":"https://consensus.app","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free unlimited; Pro $8.99/mo"},
{"name":"Elicit","category":"Research & Academia","description":"research paper literature review science extract data academic","url":"https://elicit.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"5000 credits; Plus $10/mo"},
{"name":"Scholarcy","category":"Research & Academia","description":"summarize paper research flashcard reference extraction academic","url":"https://scholarcy.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free extension; $9.99/mo"},
{"name":"Research Rabbit","category":"Research & Academia","description":"academic paper graph literature discover citation map network","url":"https://researchrabbit.ai","pricing_model":"Free","rating":"4.9/5","pricing_details":"100% free"},
{"name":"SciSpace","category":"Research & Academia","description":"research paper pdf science read literature explain copilot","url":"https://typeset.io","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free basic; Premium $12/mo"},
{"name":"Wolfram Alpha","category":"Math & Computation","description":"math calculation logic science data knowledge engine compute","url":"https://wolframalpha.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Pro $5/mo"},
{"name":"Photomath","category":"Math & Homework","description":"math equation solve picture homework steps calculator camera","url":"https://photomath.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free basic; Plus $9.99/mo"},
{"name":"Socratic","category":"Homework & Study","description":"homework math science history study google help student learning","url":"https://socratic.org","pricing_model":"Free","rating":"4.6/5","pricing_details":"100% free app"},
{"name":"Duolingo Max","category":"Language Learning","description":"learn language french spanish italian tutor education practice roleplay","url":"https://duolingo.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"~$29.99/mo or $167.99/yr"},
{"name":"DeepL","category":"Translation","description":"translate translation language text french english grammar neural","url":"https://deepl.com","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free; Pro $8.74/user/mo"},
{"name":"Khanmigo","category":"Tutoring & Education","description":"tutoring khan academy education math science student help guide","url":"https://khanmigo.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free US educators; $4/mo"},
{"name":"MagicSchool AI","category":"Education Tools","description":"teacher lesson plan worksheet education classroom assignment","url":"https://magicschool.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Plus $8.33/mo"},
{"name":"Eduaide.ai","category":"Education Tools","description":"teaching resource lesson plan assignment education content","url":"https://eduaide.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free limits; Pro $5.99/mo"},
{"name":"Diffit","category":"Education Tools","description":"teacher resource lesson differentiation reading level adapt","url":"https://diffit.me","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free basic; Pro ~$14.99/mo"},
{"name":"Curipod","category":"Education Tools","description":"interactive lesson presentation slide education classroom poll quiz","url":"https://curipod.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Premium $7.50/mo"},
{"name":"QuestionWell","category":"Education Tools","description":"quiz question generator assessment education test multiple choice","url":"https://questionwell.org","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; Teacher $7/mo"},
{"name":"Mathway","category":"Math & Homework","description":"math solver algebra calculus geometry step solution problem","url":"https://mathway.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free answers; Steps $9.99/mo"},
{"name":"Symbolab","category":"Math & Computation","description":"math solver calculus algebra equation graph step calculator","url":"https://symbolab.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; Pro $2.49/wk or $6.99/mo"},
{"name":"Connected Papers","category":"Research & Academia","description":"academic paper visualization graph citation relation visual map","url":"https://connectedpapers.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"5 graphs/mo; Academic $3/mo"},
{"name":"Litmaps","category":"Research & Academia","description":"literature map research paper citation visual discover timeline","url":"https://litmaps.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free basic; Pro $10/mo"},
{"name":"Scinapse","category":"Research & Academia","description":"academic search engine paper find research science journal","url":"https://scinapse.io","pricing_model":"Free","rating":"4.5/5","pricing_details":"Free to search"},
{"name":"Paperpal","category":"Academic Writing","description":"academic writing paper edit proofread manuscript research journal","url":"https://paperpal.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"200 edits; Prime $12/mo"},
{"name":"Trinka AI","category":"Academic Writing","description":"academic writing grammar check science paper correction technical","url":"https://trinka.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"10000 words; Premium $6.67/mo"},
{"name":"Writefull","category":"Academic Writing","description":"academic writing language edit paper research paraphrase scientific","url":"https://writefull.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; Premium $5.46/mo"},
{"name":"GeoGebra","category":"Math & Computation","description":"math geometry algebra graph interactive visual calculator dynamic","url":"https://geogebra.org","pricing_model":"Free","rating":"4.8/5","pricing_details":"100% free"},
{"name":"Luma AI","category":"3D Generation","description":"3d capture scan model object video nerf point cloud photogrammetry","url":"https://lumalabs.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; paid $7.99-$30/mo"},
{"name":"Spline AI","category":"3D Design","description":"3d design model render web browser interactive scene prompt","url":"https://spline.design","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Super $9/mo"},
{"name":"CSM AI","category":"3D Generation","description":"3d model generation mesh video image to 3d common sense machines","url":"https://csm.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; Pro ~$20/mo"},
{"name":"Meshy","category":"3D Generation","description":"3d model text to 3d mesh texture generation asset game","url":"https://meshy.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free credits; Pro $20/mo"},
{"name":"Rosebud AI","category":"Game Development","description":"game development asset code sprite visual create browser","url":"https://rosebud.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free basic; Pro available"},
{"name":"Scenario","category":"Game Assets","description":"game asset image texture 2d 3d style consistent train","url":"https://scenario.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Pro $20/mo"},
{"name":"Promethean AI","category":"Game Development","description":"game environment 3d asset art pipeline management virtual world","url":"https://prometheanai.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free non-commercial; $10/mo"},
{"name":"Tripo3D","category":"3D Generation","description":"text to 3d model generation fast mesh asset game-ready","url":"https://tripo3d.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free daily; Pro $10/mo"},
{"name":"3DFY AI","category":"3D Generation","description":"text to 3d model generation quality asset create clean mesh","url":"https://3dfy.ai","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free credits; $15/mo"},
{"name":"Sloyd","category":"3D Generation","description":"3d model generation parametric asset game prop fast low-poly","url":"https://sloyd.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free; Pro $15/mo"},
{"name":"Rodin","category":"3D Generation","description":"3d model generation high quality mesh sculpt create deemos","url":"https://deemos.com/rodin","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free credits; subscriptions"},
{"name":"Polycam","category":"3D Scanning","description":"3d scan capture photo to model photogrammetry lidar phone","url":"https://poly.cam","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Pro $14.99/mo"},
{"name":"Inworld AI","category":"Game NPC","description":"game npc character ai personality dialogue behavior npc","url":"https://inworld.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free trial; API $20/mo"},
{"name":"Convai","category":"Game NPC","description":"game npc character conversation dialogue intelligence voice","url":"https://convai.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Growth $29/mo"},
{"name":"Charisma AI","category":"Game Narrative","description":"game character story narrative dialogue interactive fiction branching","url":"https://charisma.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free engine; commercial license"},
{"name":"Layer AI","category":"Game Assets","description":"game art asset creation 2d 3d texture style consistent","url":"https://layer.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Pro ~$20/mo"},
{"name":"Kaedim","category":"3D Generation","description":"2d image to 3d model generation mesh automatic rigged","url":"https://kaedim.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"7-day $50 trial; Indie $400/mo"},
{"name":"Masterpiece X","category":"3D Generation","description":"3d model generation rigged animated character vr ready","url":"https://masterpiecex.com","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free credits; packs"},
{"name":"Julius AI","category":"Data Analysis","description":"data analysis spreadsheet csv python statistics visualization chat","url":"https://julius.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"15 queries; Plus $35/mo"},
{"name":"Akkio","category":"Predictive Analytics","description":"predictive analytics machine learning data forecast business no-code","url":"https://akkio.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"14-day trial; $49/mo"},
{"name":"Obviously AI","category":"Predictive Analytics","description":"predictive analytics data science no code machine learning train","url":"https://obviously.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"~$99/mo after trial"},
{"name":"DataRobot","category":"Machine Learning","description":"automl machine learning model auto build deploy enterprise","url":"https://datarobot.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom pricing"},
{"name":"H2O.ai","category":"Machine Learning","description":"machine learning open source auto model enterprise ai platform","url":"https://h2o.ai","pricing_model":"Free","rating":"4.5/5","pricing_details":"Free OSS; Enterprise quote"},
{"name":"Tableau Pulse","category":"Business Intelligence","description":"business intelligence dashboard data visualization analytics metrics","url":"https://tableau.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$15-$75/user/mo"},
{"name":"Power BI Copilot","category":"Business Intelligence","description":"business intelligence dashboard microsoft data analytics report dax","url":"https://powerbi.microsoft.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$10/user/mo"},
{"name":"ThoughtSpot Sage","category":"Business Intelligence","description":"search analytics data question natural language business intelligence","url":"https://thoughtspot.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"~$95/mo"},
{"name":"MonkeyLearn","category":"Text Analytics","description":"text analysis nlp sentiment classification data mining no-code","url":"https://monkeylearn.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Teams $299/mo"},
{"name":"Rows","category":"Spreadsheets","description":"spreadsheet ai data analysis formula import web scrape modern","url":"https://rows.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Plus $15/mo"},
{"name":"Polymer Search","category":"Data Visualization","description":"data visualization spreadsheet dashboard analytics search interactive","url":"https://polymersearch.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"14-day trial; $20/mo"},
{"name":"PandasAI","category":"Data Analysis","description":"python pandas data analysis code natural language query dataframe","url":"https://pandas-ai.com","pricing_model":"Free","rating":"4.6/5","pricing_details":"Free open source"},
{"name":"ChatCSV","category":"Data Analysis","description":"csv data chat question spreadsheet analyze file conversational","url":"https://chatcsv.co","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free trial; $10/mo"},
{"name":"ExcelFormulaBot","category":"Spreadsheets","description":"excel formula spreadsheet formula generate vba code google sheets","url":"https://excelformulabot.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"5 formulas/mo; $6.99/mo"},
{"name":"Ajelix","category":"Spreadsheets","description":"excel spreadsheet formula vba macro automation data productivity","url":"https://ajelix.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free limits; $5.95/mo"},
{"name":"DataLab","category":"Data Science","description":"data science notebook python analysis code collaborate datacamp","url":"https://datalab.to","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Premium $9/mo"},
{"name":"Hex AI","category":"Data Science","description":"data science notebook sql python analytics collaborate workspace","url":"https://hex.tech","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free community; Teams $36/user/mo"},
{"name":"Deepnote","category":"Data Science","description":"data science notebook python jupyter collaborate analysis cloud","url":"https://deepnote.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Teams $39/editor/mo"},
{"name":"MarketMuse","category":"SEO & Content","description":"content strategy seo topic cluster research optimization depth","url":"https://marketmuse.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"10 queries; $149/mo"},
{"name":"Clearscope","category":"SEO & Content","description":"seo content optimization keyword editor ranking article grading","url":"https://clearscope.io","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$170/mo for 20 briefs"},
{"name":"SEMrush AI","category":"SEO & Marketing","description":"seo keyword research competitor analysis traffic ranking digital marketing","url":"https://semrush.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"7-day trial; Pro $139.95/mo"},
{"name":"Ahrefs AI","category":"SEO & Marketing","description":"seo backlink keyword research competitor audit ranking site explorer","url":"https://ahrefs.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Lite $99/mo"},
{"name":"Alli AI","category":"SEO Automation","description":"seo automation optimize page rank site audit automate","url":"https://alliai.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"10-day trial; $299/mo"},
{"name":"GrowthBar","category":"SEO & Marketing","description":"seo chrome extension keyword competitor research marketing content","url":"https://growthbarseo.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"5-day trial; $36/mo"},
{"name":"NeuralText","category":"SEO & Content","description":"seo content writing keyword research optimization article brief","url":"https://neuraltext.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free; Starter $19/mo"},
{"name":"Outranking","category":"SEO & Content","description":"seo content brief article optimization ranking strategy research","url":"https://outranking.io","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$29 first mo then $79/mo"},
{"name":"Scalenut","category":"SEO & Content","description":"seo content marketing strategy research writing optimization nlp","url":"https://scalenut.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"7-day trial; $19/mo"},
{"name":"AdCreative AI","category":"Ad Creative","description":"ad creative banner design marketing conversion ad copy generation","url":"https://adcreative.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"7-day trial; $29/mo"},
{"name":"Pencil AI","category":"Ad Creative","description":"ad creative brand marketing video social media campaign ecommerce","url":"https://trypencil.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"~$119/mo"},
{"name":"Smartwriter AI","category":"Cold Outreach","description":"cold email outreach personalization backlink marketing linkedin","url":"https://smartwriter.ai","pricing_model":"Paid","rating":"4.3/5","pricing_details":"7-day trial; $49/mo"},
{"name":"Lavender","category":"Email Optimization","description":"sales email coach optimize cold outreach reply rate grade","url":"https://lavender.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"5 emails/mo; Pro $29/mo"},
{"name":"Lyne AI","category":"Cold Outreach","description":"cold email personalization outreach sales lead research opening","url":"https://lyne.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"25 credits/mo; $39/mo"},
{"name":"Regie.ai","category":"Sales Sequences","description":"sales content sequence email outreach marketing automation sequence","url":"https://regie.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free extension; Pro $59/user/mo"},
{"name":"Optimo","category":"Marketing Automation","description":"marketing task automation seo social media content ads free","url":"https://askoptimo.com","pricing_model":"Free","rating":"4.5/5","pricing_details":"100% free"},
{"name":"Brand24 AI","category":"Social Listening","description":"social media monitoring brand mention sentiment reputation tracking","url":"https://brand24.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"14-day trial; $79/mo"},
{"name":"Meltwater AI","category":"Media Monitoring","description":"media monitoring social listening pr reputation analytics enterprise","url":"https://meltwater.com","pricing_model":"Paid","rating":"4.3/5","pricing_details":"Enterprise custom"},
{"name":"Apollo.io","category":"Sales Intelligence","description":"sales lead prospect b2b contact database outreach email","url":"https://apollo.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free unlimited email; Basic $49/user/mo"},
{"name":"Seamless.ai","category":"Sales Intelligence","description":"sales lead prospect contact database b2b search engine real-time","url":"https://seamless.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"50 credits; custom Pro"},
{"name":"ZoomInfo Copilot","category":"Sales Intelligence","description":"sales intelligence b2b data contact company prospect intent","url":"https://zoominfo.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Free trial; annual custom"},
{"name":"Gong AI","category":"Revenue Intelligence","description":"sales call recording conversation intelligence revenue analysis pipeline","url":"https://gong.io","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Annual custom per team"},
{"name":"Chorus.ai","category":"Revenue Intelligence","description":"sales conversation intelligence call recording analysis coaching","url":"https://chorus.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom per user base"},
{"name":"Salesloft AI","category":"Sales Engagement","description":"sales engagement outreach email cadence conversation dialer","url":"https://salesloft.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom packages"},
{"name":"Outreach AI","category":"Sales Engagement","description":"sales engagement sequence email cadence pipeline management execution","url":"https://outreach.io","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Enterprise custom"},
{"name":"HubSpot AI","category":"CRM & Marketing","description":"crm sales marketing customer service hub inbound automation","url":"https://hubspot.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free CRM; Starter $15/seat/mo"},
{"name":"Salesforce Einstein","category":"CRM & Analytics","description":"crm sales cloud customer relationship prediction lead scoring","url":"https://salesforce.com/einstein","pricing_model":"Paid Add-on","rating":"4.4/5","pricing_details":"~$50-$75/user/mo"},
{"name":"Zoho Zia","category":"CRM Assistant","description":"crm sales assistant zoho prediction lead scoring anomaly","url":"https://zoho.com/zia","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Included in Zoho paid $14/user/mo+"},
{"name":"Freshworks Freddy AI","category":"CRM & Support","description":"crm sales customer support freshdesk freshsales automate","url":"https://freshworks.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; AI add-ons"},
{"name":"Fireflies.ai","category":"Meeting Intelligence","description":"meeting transcription notes sales call recording summarize search","url":"https://fireflies.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free limited; Pro $10/user/mo"},
{"name":"Avoma AI","category":"Meeting Intelligence","description":"meeting assistant sales coaching conversation intelligence notes analysis","url":"https://avoma.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free planner; Starter $19/user/mo"},
{"name":"Balto","category":"Real-time Coaching","description":"real time sales coaching call guidance script live objection","url":"https://balto.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per call volume"},
{"name":"Findem","category":"Talent Intelligence","description":"talent data people search recruiting sourcing candidate 3d data","url":"https://findem.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom"},
{"name":"Intercom Fin","category":"Customer Support AI","description":"customer support chatbot help desk ticket automation autonomous resolution","url":"https://intercom.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$29/seat/mo; Fin $0.99/resolution"},
{"name":"Zendesk AI","category":"Customer Support AI","description":"customer support ticket help desk automation chatbot triage sentiment","url":"https://zendesk.com","pricing_model":"Paid Add-on","rating":"4.5/5","pricing_details":"Suite Team $55/agent/mo; AI $50 add-on"},
{"name":"Freshdesk AI","category":"Customer Support AI","description":"customer support ticket help desk automation freshworks freddy bot","url":"https://freshdesk.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free 10 agents; Growth $15/agent/mo"},
{"name":"Help Scout AI","category":"Customer Support AI","description":"customer support email help desk shared inbox draft reply summarize","url":"https://helpscout.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"15-day trial; $20/user/mo"},
{"name":"Ada","category":"Customer Support AI","description":"customer service chatbot automation brand resolution autonomous","url":"https://ada.cx","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per resolution volume"},
{"name":"LivePerson","category":"Conversational Commerce","description":"customer messaging conversational commerce live chat enterprise","url":"https://liveperson.com","pricing_model":"Paid","rating":"4.2/5","pricing_details":"Custom contracts"},
{"name":"Drift","category":"Conversational Marketing","description":"conversational marketing chatbot sales engagement qualify lead booking","url":"https://drift.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Premium $2,500/mo"},
{"name":"Manychat","category":"Chatbot Automation","description":"facebook messenger chatbot instagram automation marketing sales","url":"https://manychat.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free 1000 contacts; Pro $15/mo"},
{"name":"Chatfuel","category":"Chatbot Automation","description":"facebook messenger chatbot instagram bot automation sales flow","url":"https://chatfuel.com","pricing_model":"Paid","rating":"4.3/5","pricing_details":"7-day trial; Business $14.39/mo"},
{"name":"Tars","category":"Chatbot Automation","description":"chatbot landing page lead generation conversational form","url":"https://hellotars.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"14-day trial; $99/mo"},
{"name":"Landbot","category":"Chatbot Automation","description":"conversational chatbot landing page whatsapp lead no-code","url":"https://landbot.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free sandbox; Starter €40/mo"},
{"name":"Yellow.ai","category":"Enterprise Chatbot","description":"conversational ai customer support automation enterprise dynamic agent","url":"https://yellow.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free 100 MTUs; custom enterprise"},
{"name":"Kore.ai","category":"Enterprise Chatbot","description":"enterprise chatbot conversational virtual assistant platform","url":"https://kore.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per scale"},
{"name":"Rasa","category":"Chatbot Framework","description":"open source conversational ai chatbot framework nlp custom","url":"https://rasa.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free OSS; Enterprise on request"},
{"name":"Forethought","category":"Customer Support AI","description":"customer support ai ticket resolution triage automate generative","url":"https://forethought.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per ticket volume"},
{"name":"Solvvy","category":"Customer Support AI","description":"customer support self service chatbot resolution knowledge base zoom","url":"https://solvvy.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom via Zoom"},
{"name":"Eightfold AI","category":"Talent Intelligence","description":"talent intelligence recruiting hiring candidate matching diversity","url":"https://eightfold.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"~$7-$10 PEPM"},
{"name":"HireVue","category":"Video Interviewing","description":"video interview candidate assessment hiring screening structured","url":"https://hirevue.com","pricing_model":"Paid","rating":"4.2/5","pricing_details":"~$35,000/yr Essentials"},
{"name":"Fetcher","category":"Recruiting Automation","description":"recruiting sourcing candidate email automation outreach sourcing","url":"https://fetcher.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per positions"},
{"name":"SeekOut","category":"Talent Search","description":"talent search sourcing candidate recruiting people database diversity","url":"https://seekout.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per seats"},
{"name":"Textio","category":"Inclusive Writing","description":"job description writing bias inclusive language recruiting","url":"https://textio.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Custom per headcount"},
{"name":"HireEZ","category":"Outbound Recruiting","description":"recruiting sourcing candidate outbound talent search aggregator","url":"https://hireez.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per seats"},
{"name":"Gem","category":"Recruiting CRM","description":"recruiting candidate relationship talent pipeline sourcing crm","url":"https://gem.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Annual per team size"},
{"name":"Paradox","category":"Recruiting Assistant","description":"recruiting assistant chatbot olivia scheduling hiring automation","url":"https://paradox.ai","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Custom per hiring volume"},
{"name":"Manatal","category":"Applicant Tracking","description":"recruiting software ats candidate management hiring social media","url":"https://manatal.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"14-day trial; $15/user/mo"},
{"name":"Humanly","category":"Recruiting Automation","description":"recruiting candidate screening chatbot interview scheduling automation","url":"https://humanly.io","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per volume"},
{"name":"Vervoe","category":"Skills Assessment","description":"skills assessment candidate test hiring evaluation performance auto-grade","url":"https://vervoe.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Free trial; credit or annual"},
{"name":"Interviewer.AI","category":"Video Interviewing","description":"video interview pre screening candidate assessment hiring structured","url":"https://interviewer.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free trial; ~$19/mo"},
{"name":"Retorio","category":"Behavioral Assessment","description":"behavioral assessment video analysis hiring candidate personality","url":"https://retorio.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom per volume"},
{"name":"Torque AI","category":"People Analytics","description":"workforce analytics hr data people intelligence retention forecast","url":"https://torquehq.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise custom"},
{"name":"Vic.ai","category":"Accounts Payable","description":"accounts payable invoice processing automation finance autonomous","url":"https://vic.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Usage-based per invoice volume"},
{"name":"Ramp AI","category":"Expense Management","description":"corporate card expense management finance spend automation categorize","url":"https://ramp.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Plus $12/user/mo"},
{"name":"Brex AI","category":"Expense Management","description":"corporate card expense finance startup spend management budget","url":"https://brex.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free essential; Enterprise custom"},
{"name":"Expensify AI","category":"Expense Management","description":"expense report receipt scan reimbursement finance smartscan","url":"https://expensify.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; $5/user/mo"},
{"name":"Botkeeper","category":"Bookkeeping","description":"bookkeeping accounting automation tax ledger reconcile","url":"https://botkeeper.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom per client volume"},
{"name":"Docyt","category":"Accounting Automation","description":"accounting automation bookkeeping receipt document finance ledger sync","url":"https://docyt.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$299/mo small business"},
{"name":"Trullion","category":"Accounting & Audit","description":"accounting lease revenue recognition audit finance workflow","url":"https://trullion.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per contract volume"},
{"name":"FloQast AI","category":"Month-end Close","description":"accounting close reconciliation finance month end compliance","url":"https://floqast.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per team"},
{"name":"BlackLine AI","category":"Financial Close","description":"accounting reconciliation finance close automation enterprise intercompany","url":"https://blackline.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom per modules"},
{"name":"DataRails","category":"FP&A","description":"excel spreadsheet finance planning analysis fp&a consolidate","url":"https://datarails.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"~$24,000/yr"},
{"name":"Domo AI","category":"Business Intelligence","description":"business intelligence finance dashboard analytics data real-time","url":"https://domo.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free small teams; $300/mo+"},
{"name":"Cleo AI","category":"Personal Finance","description":"personal finance budget spending money chatbot assistant conversational","url":"https://web.meetcleo.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Plus $5.99/mo"},
{"name":"Monarch Money AI","category":"Personal Finance","description":"personal finance budget tracking money management net worth","url":"https://monarchmoney.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$14.99/mo or $99.99/yr"},
{"name":"Rocket Money AI","category":"Personal Finance","description":"personal finance subscription cancel budget save money bills","url":"https://rocketmoney.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Premium $6-$12/mo"},
{"name":"Wealthfront AI","category":"Robo-Advisor","description":"investing portfolio robo advisor automated finance index fund tax-loss","url":"https://wealthfront.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"0.25% annual fee"},
{"name":"Betterment AI","category":"Robo-Advisor","description":"investing robo advisor portfolio automated retirement goal planning","url":"https://betterment.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"0.25% annual or $4/mo"},
{"name":"Harvey AI","category":"Legal AI","description":"legal research contract analysis law firm attorney assistant llm","url":"https://harvey.ai","pricing_model":"Paid","rating":"4.9/5","pricing_details":"Enterprise law firm pricing"},
{"name":"Lexis+ AI","category":"Legal Research","description":"legal research case law statute search lexisnexis generative","url":"https://lexisnexis.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom firm add-on"},
{"name":"CoCounsel","category":"Legal AI","description":"legal ai assistant research contract review casetext document review","url":"https://casetext.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Custom per firm"},
{"name":"Kira Systems","category":"Contract Analysis","description":"contract analysis review extraction due diligence m&a machine learning","url":"https://kirasystems.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom annual per volume"},
{"name":"Luminance","category":"Contract Analysis","description":"contract review legal document analysis due diligence negotiation","url":"https://luminance.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom"},
{"name":"Lawgeex","category":"Contract Review","description":"contract review automation legal approval workflow playbook","url":"https://lawgeex.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Annual per throughput"},
{"name":"LegalZoom AI","category":"Legal Services","description":"legal document formation llc incorporation trademark online","url":"https://legalzoom.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free LLC filing + state fees; ~$299/yr"},
{"name":"Spellbook","category":"Contract Drafting","description":"contract drafting legal word addin lawyer review suggestion terms","url":"https://spellbook.legal","pricing_model":"Paid","rating":"4.8/5","pricing_details":"7-day trial; ~$89/mo"},
{"name":"Lex Machina","category":"Legal Analytics","description":"legal analytics case law data litigation strategy judge ruling","url":"https://lexmachina.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per practice area"},
{"name":"DoNotPay","category":"Consumer Legal","description":"legal chatbot consumer rights fight tickets refund sue small claims","url":"https://donotpay.com","pricing_model":"Paid","rating":"4.1/5","pricing_details":"$36/quarter ($12/mo)"},
{"name":"OneTrust AI","category":"Privacy & Compliance","description":"privacy compliance gdpr data governance risk ccpa enterprise","url":"https://onetrust.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom per modules"},
{"name":"Drata AI","category":"Security Compliance","description":"compliance automation soc2 iso27001 security audit continuous","url":"https://drata.com","pricing_model":"Paid","rating":"4.9/5","pricing_details":"Custom per frameworks"},
{"name":"Vanta AI","category":"Security Compliance","description":"compliance security automation soc2 trust audit hipaa","url":"https://vanta.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Custom per workforce size"},
{"name":"Secureframe AI","category":"Security Compliance","description":"compliance security certification soc2 hipaa iso27001 pci audit","url":"https://secureframe.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Custom per scope"},
{"name":"Glass Health","category":"Clinical Decision Support","description":"clinical diagnosis medical decision support physician differential","url":"https://glass.health","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Pro $10/mo"},
{"name":"Hippocratic AI","category":"Healthcare AI","description":"healthcare patient safety medical chatbot non-diagnostic follow-up outreach","url":"https://hippocraticai.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise per deployment"},
{"name":"Nabla","category":"Medical Scribe","description":"medical transcription clinical notes physician assistant ambient scribe","url":"https://nabla.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Pro $119/physician/mo"},
{"name":"Suki AI","category":"Medical Voice Assistant","description":"voice assistant physician clinical notes medical documentation ehr","url":"https://suki.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"~$199/user/mo"},
{"name":"Augmedix","category":"Medical Documentation","description":"medical documentation clinical notes ambient scribe clinician conversation","url":"https://augmedix.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Custom per network"},
{"name":"Nuance DAX","category":"Medical Documentation","description":"medical voice documentation clinical ambient recording microsoft dragon","url":"https://nuance.com/dax","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise per physician"},
{"name":"Abridge","category":"Medical Documentation","description":"medical conversation summary patient notes health record soap clinical","url":"https://abridge.com","pricing_model":"Paid","rating":"4.9/5","pricing_details":"Enterprise per license"},
{"name":"DeepScribe","category":"Medical Scribe","description":"medical scribe ai clinical documentation physician notes ehr ambient","url":"https://deepscribe.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"~$250-$300/physician/mo"},
{"name":"Buoy Health","category":"Symptom Checker","description":"symptom checker health assessment triage chatbot navigation","url":"https://buoyhealth.com","pricing_model":"Free","rating":"4.5/5","pricing_details":"Free consumer; enterprise"},
{"name":"Ada Health","category":"Symptom Checker","description":"symptom assessment health checker personal medical condition","url":"https://ada.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"100% free consumer"},
{"name":"K Health","category":"Telemedicine","description":"virtual doctor telemedicine symptom chat primary care ai","url":"https://khealth.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$49 visit; $29/mo membership"},
{"name":"PathAI","category":"Pathology AI","description":"pathology medical diagnosis tissue analysis cancer precision","url":"https://pathai.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise per deployment"},
{"name":"Tempus AI","category":"Precision Medicine","description":"precision medicine genomic data cancer oncology clinical trial","url":"https://tempus.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Institutional pricing"},
{"name":"Paige AI","category":"Digital Pathology","description":"pathology digital diagnosis cancer tissue prostate fda-cleared","url":"https://paige.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise per lab"},
{"name":"Figma AI","category":"Design Tools","description":"ui ux design figma interface prototype component generate layout","url":"https://figma.com","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free; Pro $12/editor/mo"},
{"name":"Visily","category":"Wireframing","description":"wireframe mockup ui design app prototype sketch screenshot","url":"https://visily.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Pro ~$9/user/mo"},
{"name":"Whimsical AI","category":"Visual Thinking","description":"wireframe mind map flowchart diagram visual thinking ai","url":"https://whimsical.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"3 boards; Pro $10/editor/mo"},
{"name":"Miro AI","category":"Whiteboard","description":"whiteboard brainstorm mind map diagram sticky note cluster","url":"https://miro.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"3 boards; Starter $8/member/mo"},
{"name":"Framer AI","category":"Website Builder","description":"website design landing page web builder prototype responsive","url":"https://framer.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Mini $5/mo; Basic $15/mo"},
{"name":"Webflow AI","category":"Website Builder","description":"website builder no code web design responsive cms visual","url":"https://webflow.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Basic $14/mo"},
{"name":"Khroma","category":"Color Palettes","description":"color palette design scheme brand generate favorite personalized","url":"https://khroma.co","pricing_model":"Free","rating":"4.6/5","pricing_details":"100% free"},
{"name":"Fontjoy","category":"Typography","description":"font pairing typography design text web heading body deep learning","url":"https://fontjoy.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"100% free open source"},
{"name":"Coolors AI","category":"Color Palettes","description":"color palette generator scheme design brand gradient extract","url":"https://coolors.co","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free generator; Pro $3/mo"},
{"name":"Adobe Color AI","category":"Color Palettes","description":"color wheel palette scheme design extract gradient adobe sensei","url":"https://color.adobe.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free web tool"},
{"name":"Looka AI","category":"Logo Design","description":"logo design brand identity maker business create generate","url":"https://looka.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free design; $20 basic"},
{"name":"Tailor Brands AI","category":"Logo & Branding","description":"logo brand identity design business llc formation domain","url":"https://tailorbrands.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free logo; $3.99/mo"},
{"name":"Brandmark AI","category":"Logo Design","description":"logo design brand identity visual create business vector svg","url":"https://brandmark.io","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$25-$65 one-time"},
{"name":"DesignEvo AI","category":"Logo Design","description":"logo maker design icon brand create template 10000 fonts","url":"https://designevo.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free low-res; $24.99 basic"},
{"name":"Mobbin AI","category":"Design Reference","description":"mobile app design reference screenshot pattern library ui","url":"https://mobbin.com","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free limited; Pro $10/mo"},
{"name":"Architechtures","category":"Architecture Design","description":"architectural design residential building floor plan generation","url":"https://architechtures.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Pro ~€50/mo"},
{"name":"TestFit","category":"Real Estate Feasibility","description":"site planning feasibility real estate development layout parking","url":"https://testfit.io","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Custom per team"},
{"name":"Autodesk Forma","category":"Architecture Design","description":"urban planning site design building density wind noise sun","url":"https://autodesk.com/forma","pricing_model":"Paid","rating":"4.6/5","pricing_details":"AEC collection; ~$185/mo"},
{"name":"Finch 3D","category":"Architecture Design","description":"architectural floor plan design optimization building parametric","url":"https://finch3d.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"€99/user/mo"},
{"name":"Maket.ai","category":"Architecture Design","description":"residential floor plan generation architectural design zoning","url":"https://maket.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Pro $24/mo"},
{"name":"Archistar","category":"Property Intelligence","description":"property development site analysis zoning feasibility data","url":"https://archistar.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per region"},
{"name":"Interior AI","category":"Interior Design","description":"interior design room decoration home renovation style virtual staging","url":"https://interiorai.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free trial; Pro $29/mo"},
{"name":"RoomGPT","category":"Interior Design","description":"room redesign interior decoration photo to design style","url":"https://roomgpt.io","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free credits; $9 packs"},
{"name":"Reimagine Home AI","category":"Interior Design","description":"home interior redesign renovation virtual staging exterior curb","url":"https://reimaginehome.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; Pro $14/mo"},
{"name":"Decorilla AI","category":"Interior Design","description":"interior design virtual room decoration 3d visualize consultation","url":"https://decorilla.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$549/room"},
{"name":"Cedreo AI","category":"Home Design","description":"home design floor plan 3d render architectural builder","url":"https://cedreo.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"1 project free; $119/mo"},
{"name":"PromeAI","category":"Architectural Rendering","description":"architectural render design sketch to render building texture","url":"https://promeai.pro","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free coins; Base $19/mo"},
{"name":"Cala","category":"Fashion Design","description":"fashion design apparel brand production supply chain tech pack","url":"https://ca.la","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free design; custom production"},
{"name":"Resleeve","category":"Fashion Design","description":"fashion design sketch garment apparel create ai photorealistic","url":"https://resleeve.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; Premium $33/mo"},
{"name":"VModel AI","category":"Fashion Photography","description":"fashion model photography product apparel ecommerce mannequin","url":"https://vmodel.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; ~$19/mo"},
{"name":"Botika","category":"Fashion Photography","description":"fashion model photo apparel ecommerce product image generation","url":"https://botika.io","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free credits; Starter $15/mo"},
{"name":"Vue AI Fashion","category":"Fashion Retail","description":"fashion retail ecommerce product tagging merchandising personalization","url":"https://vue.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise custom"},
{"name":"Designovel","category":"Fashion Design","description":"fashion design garment pattern trend apparel create data","url":"https://designovel.com","pricing_model":"Paid","rating":"4.4/5","pricing_details":"Enterprise custom"},
{"name":"Heuritech","category":"Fashion Trends","description":"fashion trend forecasting analysis data prediction social media","url":"https://heuritech.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise annual"},
{"name":"Stylumia AI","category":"Fashion Trends","description":"fashion demand forecasting consumer trend analytics retail","url":"https://stylumia.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom"},
{"name":"Intelistyle AI","category":"Fashion Styling","description":"fashion styling outfit recommendation personalization virtual stylist","url":"https://intelistyle.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise per traffic"},
{"name":"ChefGPT","category":"Food & Recipes","description":"recipe cooking meal plan ingredient food suggestion kitchen","url":"https://chefgpt.app","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Basic $2.99/mo"},
{"name":"DishGen","category":"Food & Recipes","description":"recipe generator ingredient meal cooking idea food unique","url":"https://dishgen.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free daily; Premium $3.99/mo"},
{"name":"Let's Foodie","category":"Food & Recipes","description":"recipe ingredient cooking meal food idea kitchen leftover","url":"https://letsfoodie.com","pricing_model":"Free","rating":"4.4/5","pricing_details":"100% free web"},
{"name":"Mr. Cook","category":"Food & Recipes","description":"recipe generator meal plan cooking food ingredient shopping list","url":"https://mrcook.app","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Pro $2.99/mo"},
{"name":"SuperCook","category":"Food & Recipes","description":"recipe finder ingredient search cooking meal kitchen home","url":"https://supercook.com","pricing_model":"Free","rating":"4.8/5","pricing_details":"100% free"},
{"name":"Plant Jammer","category":"Food & Recipes","description":"vegetarian vegan recipe plant based ingredient cooking","url":"https://plantjammer.com","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free basic; premium"},
{"name":"Eat This Much AI","category":"Nutrition & Meal Planning","description":"meal plan calorie nutrition diet food automatic macro","url":"https://eatthismuch.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free basic; $9/mo"},
{"name":"Yummly AI","category":"Food & Recipes","description":"recipe search meal plan cooking food recommendation personalized","url":"https://yummly.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Pro $4.99/mo"},
{"name":"SideChef AI","category":"Food & Recipes","description":"recipe cooking step by step meal plan food guide voice","url":"https://sidechef.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Premium $4.99/mo"},
{"name":"Roam Around AI","category":"Travel Planning","description":"travel itinerary plan trip destination guide vacation day-by-day","url":"https://roamaround.io","pricing_model":"Free","rating":"4.5/5","pricing_details":"100% free"},
{"name":"Wonderplan AI","category":"Travel Planning","description":"travel plan itinerary destination trip vacation customize budget","url":"https://wonderplan.ai","pricing_model":"Free","rating":"4.6/5","pricing_details":"Free to use"},
{"name":"iPlan AI","category":"Travel Planning","description":"travel itinerary planner trip destination schedule route map","url":"https://iplan.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free app; in-app purchases"},
{"name":"Mindtrip AI","category":"Travel Planning","description":"travel assistant itinerary plan destination booking conversational","url":"https://mindtrip.ai","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free consumer"},
{"name":"Tripnotes AI","category":"Travel Planning","description":"travel notes itinerary plan destination guide trip mapping","url":"https://tripnotes.ai","pricing_model":"Free","rating":"4.6/5","pricing_details":"Free web tool"},
{"name":"Geniustour AI","category":"Travel Planning","description":"tour guide travel destination itinerary plan vacation sightseeing","url":"https://geniustour.net","pricing_model":"Free","rating":"4.3/5","pricing_details":"Free web"},
{"name":"EasyTrip AI","category":"Travel Planning","description":"travel plan trip itinerary destination vacation suggest flight hotel","url":"https://easytrip.ai","pricing_model":"Free","rating":"4.4/5","pricing_details":"Free to plan"},
{"name":"Wander AI","category":"Travel Planning","description":"travel itinerary destination plan trip suggestion vacation guide","url":"https://wanderai.app","pricing_model":"Free","rating":"4.4/5","pricing_details":"Free web app"},
{"name":"Hopper AI","category":"Travel Booking","description":"flight hotel price prediction travel booking cheap forecast","url":"https://hopper.com","pricing_model":"Free","rating":"4.8/5","pricing_details":"Free app; booking fees"},
{"name":"Kayak AI","category":"Travel Booking","description":"flight hotel car rental travel search booking compare aggregator","url":"https://kayak.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"100% free search"},
{"name":"CubiCasa AI","category":"Real Estate Tech","description":"floor plan scan property real estate measurement 3d","url":"https://cubicasa.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free 2D; $15-$35 3D"},
{"name":"Restb.ai","category":"Real Estate Tech","description":"real estate image analysis property photo listing computer vision","url":"https://restb.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"B2B API per image volume"},
{"name":"Zillow AI","category":"Real Estate Search","description":"real estate property search zestimate home value listing marketplace","url":"https://zillow.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free; agent ads"},
{"name":"HouseCanary AI","category":"Real Estate Analytics","description":"property valuation real estate analytics market data prediction","url":"https://housecanary.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom"},
{"name":"Skyline AI","category":"Real Estate Investment","description":"real estate investment property analytics market prediction commercial","url":"https://skyline.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise CRE"},
{"name":"Virtual Staging AI","category":"Real Estate Marketing","description":"virtual staging furniture interior real estate photo empty room","url":"https://virtualstagingai.app","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; $16/mo"},
{"name":"BoxBrownie AI","category":"Real Estate Marketing","description":"real estate photo editing virtual staging enhancement day-to-dusk","url":"https://boxbrownie.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$1.60-$24/image"},
{"name":"RealtyNinja AI","category":"Real Estate Marketing","description":"real estate website builder listing description property marketing","url":"https://realtyninja.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$49/mo; 14-day trial"},
{"name":"ListingAI","category":"Real Estate Marketing","description":"real estate listing description property write marketing copy","url":"https://listingai.co","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free credits; $9/mo"},
{"name":"AlphaFold","category":"Protein Structure","description":"protein structure prediction biology science deepmind fold 3d","url":"https://alphafold.com","pricing_model":"Free","rating":"4.9/5","pricing_details":"Open source 100% free"},
{"name":"ESMFold","category":"Protein Structure","description":"protein structure prediction metagenomic biology science language model","url":"https://esmfold.com","pricing_model":"Free","rating":"4.9/5","pricing_details":"Open source free"},
{"name":"IBM RXN","category":"Chemistry","description":"chemistry reaction prediction synthesis molecule drug retrosynthesis","url":"https://rxn.res.ibm.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free web platform"},
{"name":"Synthia","category":"Chemistry","description":"chemistry retrosynthesis molecule design drug synthesis route merck","url":"https://synthia.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise license"},
{"name":"Atomwise","category":"Drug Discovery","description":"drug discovery molecule screening pharmaceutical ai structure-based","url":"https://atomwise.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Partnership-based"},
{"name":"Exscientia","category":"Drug Discovery","description":"drug design pharmaceutical molecule discovery medicine precision","url":"https://exscientia.ai","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Pharma partnerships"},
{"name":"Insilico Medicine","category":"Drug Discovery","description":"drug discovery aging pharmaceutical target molecule clinical trial","url":"https://insilico.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Institutional licensing"},
{"name":"Recursion Pharma","category":"Drug Discovery","description":"drug discovery phenomics biology screening cellular disease","url":"https://recursion.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Research partnerships"},
{"name":"BenevolentAI","category":"Drug Discovery","description":"drug discovery disease target knowledge graph science biomedical","url":"https://benevolentai.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Research licensing"},
{"name":"Citrine Informatics","category":"Materials Science","description":"materials science data properties prediction alloy formulation","url":"https://citrine.io","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom"},
{"name":"Materials Project","category":"Materials Science","description":"materials database science properties computation open access","url":"https://materialsproject.org","pricing_model":"Free","rating":"4.9/5","pricing_details":"100% free open access"},
{"name":"Schrödinger AI","category":"Molecular Simulation","description":"computational chemistry drug discovery molecule simulation physics","url":"https://schrodinger.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Enterprise license"},
{"name":"BenchSci AI","category":"Biomedical Research","description":"antibody reagent search science research biomedical literature","url":"https://benchsci.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise per org"},
{"name":"Deep 6 AI","category":"Clinical Trials","description":"clinical trial patient matching research medical data nlp","url":"https://deep6.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise per health system"},
{"name":"Owkin AI","category":"Medical Research","description":"medical research drug discovery federated learning biomedical privacy","url":"https://owkin.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Research partnerships"},
{"name":"Darktrace","category":"Cybersecurity","description":"cybersecurity threat detection network anomaly self-learning autonomous","url":"https://darktrace.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise per device"},
{"name":"CrowdStrike Charlotte AI","category":"Endpoint Security","description":"cybersecurity endpoint protection threat detection cloud falcon","url":"https://crowdstrike.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Enterprise add-on"},
{"name":"Microsoft Security Copilot","category":"Security Operations","description":"security analyst assistant threat incident response soc generative","url":"https://microsoft.com/security/copilot","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$4/hour SCUs"},
{"name":"Google Security AI","category":"Cloud Security","description":"cloud security threat detection analysis mandiant sec-palm","url":"https://cloud.google.com/security/ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Google Cloud subscription"},
{"name":"SentinelOne Purple AI","category":"Endpoint Security","description":"endpoint security threat hunting autonomous response purple ai","url":"https://sentinelone.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Enterprise add-on"},
{"name":"Vectra AI","category":"Network Security","description":"threat detection network cloud hybrid attack signal intelligence","url":"https://vectra.ai","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Custom per IP/workload"},
{"name":"Tenable AI","category":"Vulnerability Management","description":"vulnerability management security scan exposure risk attack path","url":"https://tenable.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Annual per assets"},
{"name":"Wiz AI","category":"Cloud Security","description":"cloud security risk posture vulnerability scan graph analysis cspm","url":"https://wiz.io","pricing_model":"Paid","rating":"4.9/5","pricing_details":"~$15,000/yr"},
{"name":"Orca AI","category":"Cloud Security","description":"cloud security lateral movement alert vulnerability agentless","url":"https://orca.security","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Per asset/workload"},
{"name":"Sonatype AI","category":"Supply Chain Security","description":"software supply chain security open source dependency malware","url":"https://sonatype.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Per developer/seat"},
{"name":"Veracode AI","category":"Application Security","description":"application security code scan vulnerability testing fix ai","url":"https://veracode.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Per application portfolio"},
{"name":"Checkmarx AI","category":"Application Security","description":"application security code scan sast vulnerability remediation","url":"https://checkmarx.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Per developer seats"},
{"name":"Lacework AI","category":"Cloud Security","description":"cloud security posture management threat detection data-driven multicloud","url":"https://lacework.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Per cloud consumption"},
{"name":"Datadog AI","category":"Monitoring & Observability","description":"monitoring observability infrastructure metrics logs traces anomaly","url":"https://datadoghq.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Pro $15/host/mo"},
{"name":"New Relic AI","category":"Monitoring & Observability","description":"observability monitoring application performance apm grook ai","url":"https://newrelic.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"100GB/mo free; $49/core/mo"},
{"name":"Dynatrace Davis","category":"Monitoring & Observability","description":"observability monitoring application performance causal ai root cause","url":"https://dynatrace.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"~$0.08/hour"},
{"name":"PagerDuty AI","category":"Incident Management","description":"incident management oncall alert response operations automation","url":"https://pagerduty.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"5 users free; Pro $21/user/mo"},
{"name":"Incident.io","category":"Incident Management","description":"incident response slack teams automation oncall status page","url":"https://incident.io","pricing_model":"Paid","rating":"4.9/5","pricing_details":"Per responder"},
{"name":"Rootly AI","category":"Incident Management","description":"incident management response automation postmortem slack timeline","url":"https://rootly.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per responder seats"},
{"name":"Sifflet AI","category":"Data Observability","description":"data observability monitoring quality pipeline alert lineage","url":"https://sifflet.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Per data assets"},
{"name":"Monte Carlo AI","category":"Data Observability","description":"data observability quality monitoring pipeline reliability lineage","url":"https://montecarlodata.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per warehouse scale"},
{"name":"Pulumi AI","category":"Infrastructure as Code","description":"infrastructure code cloud provisioning iac deploy natural language","url":"https://pulumi.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Team per resource"},
{"name":"Terraform AI","category":"Infrastructure as Code","description":"infrastructure code cloud provisioning iac terraform hashicorp","url":"https://terraform.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free CLI; HCP per resource"},
{"name":"Mage AI","category":"Data Pipeline","description":"data pipeline etl build deploy workflow orchestration open-source","url":"https://mage.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free OSS; Pro cloud"},
{"name":"Prefect AI","category":"Workflow Orchestration","description":"data pipeline workflow orchestration etl schedule error handling","url":"https://prefect.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"$35 credit; Pro $285/mo"},
{"name":"Pinecone AI","category":"Vector Database","description":"vector database similarity search embedding index rag ai","url":"https://pinecone.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free starter; serverless"},
{"name":"Weaviate AI","category":"Vector Database","description":"vector database search graph knowledge embedding semantic open-source","url":"https://weaviate.io","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free OSS; Cloud ~$25/mo"},
{"name":"Hugging Face","category":"AI Infrastructure","description":"open source machine learning hub models datasets spaces inference","url":"https://huggingface.co","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free; Pro $9/mo; GPU endpoints"},
{"name":"Groq AI","category":"AI Inference","description":"lpu language processing unit ultra fast token generation inference","url":"https://groq.com","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free playground; API per token"},
{"name":"Together AI","category":"AI Infrastructure","description":"cloud platform training fine-tuning deploying open-source models endpoints","url":"https://together.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free credits; usage-based"},
{"name":"Replicate","category":"AI Infrastructure","description":"developer platform running open source machine learning models api","url":"https://replicate.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Pay-per-second compute"},
{"name":"Rokoko Vision AI","category":"Motion Capture","description":"motion capture animation video to 3d body tracking webcam","url":"https://rokoko.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free core; Plus $20/mo"},
{"name":"Plask Motion","category":"Motion Capture","description":"motion capture animation browser 3d character mocap video","url":"https://plask.ai","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free credits; Pro $28/mo"},
{"name":"Cascadeur AI","category":"3D Animation","description":"3d character animation physics keyframe motion realistic auto-pose","url":"https://cascadeur.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free basic; Pro $300/yr"},
{"name":"DeepMotion AI","category":"Motion Capture","description":"motion capture video to 3d animation body tracking markerless","url":"https://deepmotion.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free monthly; $15/mo"},
{"name":"Krikey AI","category":"3D Animation","description":"3d animation character create video motion capture custom avatar","url":"https://krikey.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free basic; Pro $15/mo"},
{"name":"Wonder Dynamics AI","category":"VFX & 3D","description":"video to 3d animation character vfx cgi motion capture auto composite","url":"https://wonderdynamics.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Lite $16/mo; Pro $84/mo"},
{"name":"Move AI","category":"Motion Capture","description":"motion capture video phone to 3d animation body multi-camera","url":"https://move.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Single $15/mo; Multi $365/yr"},
{"name":"Cartwheel AI","category":"3D Animation","description":"3d character animation generate motion create text to animation","url":"https://cartwheel.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free trial; Pro ~$20/mo"},
{"name":"EbSynth","category":"Style Transfer","description":"style transfer animation paint over video footage keyframe propagation","url":"https://ebsynth.com","pricing_model":"Free","rating":"4.8/5","pricing_details":"100% free download"},
{"name":"Evoto AI","category":"Photo Retouching","description":"photo editing portrait retouch batch skin enhance professional","url":"https://evoto.ai","pricing_model":"Paid","rating":"4.8/5","pricing_details":"~$0.07/credit; ~$69/1200"},
{"name":"Retouch4me","category":"Photo Retouching","description":"photo retouch portrait skin edit photoshop plugin neural network dodge burn","url":"https://retouch4.me","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$124/module lifetime"},
{"name":"Imagen AI","category":"Photo Editing","description":"photo editing cull preset batch professional photographer personal style lightroom","url":"https://imagen-ai.com","pricing_model":"Paid","rating":"4.9/5","pricing_details":"~$0.05/photo; $7/mo"},
{"name":"Aftershoot AI","category":"Photo Culling","description":"photo culling editing batch photographer workflow wedding event","url":"https://aftershoot.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Culling $120/yr; Editing $240/yr"},
{"name":"Radiant Photo AI","category":"Photo Editing","description":"photo editing enhance color light quality smart scene detection","url":"https://radiantimaginglabs.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"$159 lifetime license"},
{"name":"Perfectly Clear AI","category":"Photo Correction","description":"photo correction color light enhance batch automatic lab","url":"https://eyeq.photos","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise volume"},
{"name":"VanceAI","category":"Photo Enhancement","description":"photo enhance upscale sharpen denoise enlarge restore image","url":"https://vanceai.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free credits; $4.95/100"},
{"name":"Remini","category":"Photo Enhancement","description":"photo enhance restore old face portrait quality hd","url":"https://remini.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free trial; $6.99/wk or $9.99/mo"},
{"name":"Cutout.pro","category":"Photo Editing","description":"photo edit background remove retouch enhance portrait cutout","url":"https://cutout.pro","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"5 credits; $9.90/mo"},
{"name":"ImgLarger","category":"Image Upscaling","description":"image upscale enlarge enhance photo resolution ai upscaling","url":"https://imglarger.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"8 credits/mo; $9/mo"},
{"name":"Photoglory AI","category":"Photo Restoration","description":"old photo restore colorize repair vintage enhance black and white","url":"https://photoglory.net","pricing_model":"Paid","rating":"4.5/5","pricing_details":"$19.25 standard"},
{"name":"Filter Pixel AI","category":"Photo Culling","description":"photo culling filter selection photographer batch out-of-focus duplicate","url":"https://filterpixel.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Unlimited $9/mo"},
{"name":"Podcastle AI","category":"Podcast Creation","description":"podcast recording editing audio enhance voice create text to speech","url":"https://podcastle.ai","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Storyteller $11.99/mo"},
{"name":"Riverside FM AI","category":"Podcast Recording","description":"podcast video recording remote interview studio quality transcription","url":"https://riverside.fm","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Standard $15/mo"},
{"name":"Zencastr AI","category":"Podcast Recording","description":"podcast recording remote interview audio multi-track postproduction","url":"https://zencastr.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free; Professional $20/mo"},
{"name":"Squadcast AI","category":"Podcast Recording","description":"podcast recording remote interview studio quality cloud descript","url":"https://squadcast.fm","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free with Descript; $12/mo"},
{"name":"Wondercraft AI","category":"Podcast Creation","description":"podcast creation text to podcast voice generate audio synthetic","url":"https://wondercraft.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free trial; Creator $29/mo"},
{"name":"Snipd AI","category":"Podcast Player","description":"podcast clip snippet highlight share save note transcription player","url":"https://snipd.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free; Premium $8/mo"},
{"name":"Podsqueeze AI","category":"Podcast Content","description":"podcast content repurpose clip show notes transcript summary","url":"https://podsqueeze.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"50 min/mo; Pro $12/mo"},
{"name":"Deciphr AI","category":"Podcast Content","description":"podcast transcript show notes timestamp chapter summary content","url":"https://deciphr.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; $29/mo"},
{"name":"Resound AI","category":"Audio Editing","description":"podcast editing remove silence filler noise automate audio cleanup","url":"https://resound.fm","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; Creator $12/mo"},
{"name":"Headliner AI","category":"Audiograms","description":"podcast video clip audiogram waveform social share video","url":"https://headliner.app","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"5 videos/mo; Basic $7.99/mo"},
{"name":"Hootsuite AI","category":"Social Media Management","description":"social media management schedule post analytics dashboard owlywriter","url":"https://hootsuite.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"30-day trial; Professional $99/mo"},
{"name":"Sprout Social AI","category":"Social Media Management","description":"social media management analytics engagement scheduling sentiment","url":"https://sproutsocial.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"30-day trial; Standard $249/user/mo"},
{"name":"Buffer AI","category":"Social Media Management","description":"social media schedule post queue management analytics ai assistant","url":"https://buffer.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"3 channels; Essentials $6/channel/mo"},
{"name":"Later AI","category":"Social Media Management","description":"social media instagram schedule post visual planner caption hashtag","url":"https://later.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Trial; Starter $16.67/mo"},
{"name":"Predis.ai","category":"Social Content","description":"social media content create post video carousel creative generate","url":"https://predis.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"15 posts; Lite $29/mo"},
{"name":"Flick AI","category":"Social Media Marketing","description":"social media hashtag scheduler post idea caption marketing","url":"https://flick.social","pricing_model":"Paid","rating":"4.6/5","pricing_details":"7-day trial; Solo £11/mo"},
{"name":"Feedhive AI","category":"Social Media Management","description":"social media content schedule post ai generate queue recycling","url":"https://feedhive.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"7-day trial; Creator $19/mo"},
{"name":"Ocoya AI","category":"Social Media Design","description":"social media content design post template create schedule graphic","url":"https://ocoya.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; Bronze $15/mo"},
{"name":"Lately AI","category":"Content Repurposing","description":"social media content repurpose article video podcast to social posts","url":"https://lately.ai","pricing_model":"Paid","rating":"4.5/5","pricing_details":"7-day trial; Starter ~$29/mo"},
{"name":"ContentStudio AI","category":"Content Discovery","description":"social media content discover schedule post analytics ai writer","url":"https://contentstudio.io","pricing_model":"Paid","rating":"4.6/5","pricing_details":"14-day trial; Starter $25/mo"},
{"name":"SocialBee AI","category":"Social Media Management","description":"social media schedule post content category recycle copilot","url":"https://socialbee.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"14-day trial; Bootstrap $29/mo"},
{"name":"Publer AI","category":"Social Media Management","description":"social media schedule post manage analytics bulk ai text image","url":"https://publer.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"3 accounts; Professional $12/mo"},
{"name":"Shopify Magic","category":"E-commerce","description":"ecommerce store product description shop management generate free","url":"https://shopify.com/magic","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free with Shopify subscription"},
{"name":"Nosto AI","category":"E-commerce Personalization","description":"ecommerce personalization product recommendation shop experience","url":"https://nosto.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise per revenue"},
{"name":"Dynamic Yield AI","category":"E-commerce Personalization","description":"ecommerce personalization recommendation ab testing mastercard","url":"https://dynamicyield.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise per traffic"},
{"name":"Bloomreach AI","category":"E-commerce Search","description":"ecommerce search merchandising personalization commerce loomi","url":"https://bloomreach.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise per catalog"},
{"name":"Klevu AI","category":"E-commerce Search","description":"ecommerce search product discovery recommendation shop natural language","url":"https://klevu.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"~$199/mo small; enterprise"},
{"name":"Constructor.io AI","category":"E-commerce Search","description":"ecommerce search product discovery recommendation clickstream","url":"https://constructor.io","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per search query volume"},
{"name":"Algolia AI","category":"Search & Discovery","description":"search product discovery ecommerce recommend instant neural api","url":"https://algolia.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"10K requests free; $0.50/1K"},
{"name":"Klaviyo AI","category":"Email & SMS Marketing","description":"email marketing automation ecommerce sms personalization predictive","url":"https://klaviyo.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"250 contacts free; $20/mo"},
{"name":"Omnisend AI","category":"Email & SMS Marketing","description":"email sms marketing automation ecommerce omnichannel subject","url":"https://omnisend.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"250 contacts; Standard $16/mo"},
{"name":"Yotpo AI","category":"Reviews & Loyalty","description":"reviews rating user generated content loyalty ecommerce ecommerce","url":"https://yotpo.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free reviews; Pro $19/mo"},
{"name":"Bazaarvoice AI","category":"Reviews & UGC","description":"reviews rating user content ecommerce product social syndication","url":"https://bazaarvoice.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise per product lines"},
{"name":"Syte AI","category":"Visual Search","description":"visual search product discovery ecommerce recommendation image camera","url":"https://syte.ai","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise custom"},
{"name":"Fitbod AI","category":"Fitness & Workouts","description":"workout plan exercise fitness gym strength training personalized","url":"https://fitbod.me","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"3 workouts free; $12.99/mo"},
{"name":"Freeletics AI","category":"Fitness & Workouts","description":"workout fitness bodyweight training exercise coach adaptive","url":"https://freeletics.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; Coach ~$3.23/wk"},
{"name":"Zing Coach AI","category":"Fitness & Workouts","description":"personal trainer workout fitness exercise plan coach body composition","url":"https://zing.bodymobile.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; Premium $9.99/mo"},
{"name":"Future AI","category":"Remote Coaching","description":"personal trainer fitness coach workout remote 1on1 human trainer","url":"https://future.fit","pricing_model":"Paid","rating":"4.9/5","pricing_details":"$199/mo unlimited"},
{"name":"Tonal AI","category":"Smart Home Gym","description":"smart home gym strength training digital weight fitness adaptive resistance","url":"https://tonal.com","pricing_model":"Paid","rating":"4.9/5","pricing_details":"~$3,995 + $59/mo"},
{"name":"FitnessAI","category":"Workout Logging","description":"workout log exercise track progress weight lift optimize sets reps","url":"https://fitnessai.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"7-day trial; $4.99/wk"},
{"name":"Aaptiv AI","category":"Audio Fitness","description":"audio fitness workout class guided exercise trainer audio","url":"https://aaptiv.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"7-day trial; $14.99/mo"},
{"name":"Caliber AI","category":"Fitness & Nutrition","description":"personal training fitness coach nutrition plan workout evidence-based","url":"https://caliber.fitness","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free group; Coaching ~$200/mo"},
{"name":"Woebot","category":"Mental Health Chatbot","description":"mental health chatbot cbt therapy anxiety depression mood support","url":"https://woebothealth.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free consumer; B2B"},
{"name":"Wysa","category":"Mental Health Chatbot","description":"mental health chatbot anxiety stress cbt therapy support meditation","url":"https://wysa.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free core; Premium $11.99/mo"},
{"name":"Replika","category":"AI Companion","description":"ai companion friend emotional support chat conversation companion","url":"https://replika.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free chat; Pro $19.99/mo"},
{"name":"Headspace AI","category":"Meditation & Mindfulness","description":"meditation mindfulness stress sleep relax guided personalized","url":"https://headspace.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"14-day trial; $12.99/mo"},
{"name":"Calm AI","category":"Meditation & Sleep","description":"meditation sleep relaxation mindfulness stress anxiety personalized","url":"https://calm.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"7-day trial; $69.99/yr"},
{"name":"Spring Health AI","category":"Mental Health Benefit","description":"mental health benefit employer therapy care support precision matching","url":"https://springhealth.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Employer benefit"},
{"name":"Lyra Health AI","category":"Mental Health Benefit","description":"workplace mental health platform therapy coaching personalized matching","url":"https://lyrahealth.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Employer contract"},
{"name":"Tess AI","category":"Mental Health Chatbot","description":"psychological ai chatbot mental health coping wellness healthcare","url":"https://x2.ai","pricing_model":"Paid","rating":"4.4/5","pricing_details":"B2B per provider"},
{"name":"Happify AI","category":"Mental Wellness","description":"mental wellness game activity stress mood positive science-backed","url":"https://happify.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; Plus $14.99/mo"},
{"name":"Sanvello AI","category":"Mental Health","description":"mental health anxiety depression mood tracker cbt meditation therapy","url":"https://sanvello.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; Premium $8.99/mo"},
{"name":"Plum AI","category":"Personal Finance","description":"personal finance save invest budget money automatic round-up","url":"https://withplum.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; €3.99/mo"},
{"name":"Emma AI","category":"Personal Finance","description":"personal finance budget track subscription cancel spending aggregate","url":"https://emma-app.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free; Plus $4.99/mo"},
{"name":"Copilot Money AI","category":"Personal Finance","description":"personal finance budget tracker spending money apple ios","url":"https://copilot.money","pricing_model":"Paid","rating":"4.8/5","pricing_details":"$13/mo or $95/yr"},
{"name":"Acorns AI","category":"Micro-Investing","description":"investing micro invest spare change round up portfolio etf","url":"https://acorns.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"$3-$12/mo"},
{"name":"Stash AI","category":"Investing","description":"investing beginner stock etf portfolio fractional guided advice banking","url":"https://stash.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Growth $3/mo; Stash+ $9/mo"},
{"name":"Robinhood AI","category":"Trading","description":"stock trading invest crypto option commission free retirement","url":"https://robinhood.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trading; Gold $5/mo"},
{"name":"Public AI","category":"Investing Community","description":"stock investing community social portfolio fractional alpha","url":"https://public.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free trading; Premium $10/mo"},
{"name":"M1 Finance AI","category":"Automated Investing","description":"investing portfolio pie automate wealth management rebalance","url":"https://m1.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Plus $3/mo"},
{"name":"Personal Capital AI","category":"Wealth Management","description":"wealth management retirement net worth investment tracking dashboard","url":"https://personalcapital.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free dashboard; 0.89% AUM advisory"},
{"name":"Babbel AI","category":"Language Learning","description":"language learning spanish french german italian course dialogue speech","url":"https://babbel.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"First lesson free; ~$14.95/mo"},
{"name":"Rosetta Stone AI","category":"Language Learning","description":"language learning immersion pronunciation speaking course truaccent","url":"https://rosettastone.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"3-day trial; ~$11.99/mo or $299 lifetime"},
{"name":"Memrise AI","category":"Language Learning","description":"language learning flashcard vocabulary memorize spaced repetition native video","url":"https://memrise.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free core; Pro $14.99/mo"},
{"name":"Busuu AI","category":"Language Learning","description":"language learning course community native speaker practice review","url":"https://busuu.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; Premium $13.90/mo"},
{"name":"Lingvist AI","category":"Language Learning","description":"language learning vocabulary flashcard adaptive spaced machine learning","url":"https://lingvist.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"14-day trial; $9.99/mo"},
{"name":"Drops AI","category":"Language Learning","description":"language learning vocabulary visual word game daily 5 minute","url":"https://languagedrops.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"5 min free; Premium $13/mo"},
{"name":"Lingodeer AI","category":"Language Learning","description":"language learning asian korean japanese chinese grammar structured","url":"https://lingodeer.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free starter; Premium $14.99/mo"},
{"name":"HelloTalk AI","category":"Language Exchange","description":"language exchange practice native speaker chat correction translation","url":"https://hellotalk.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free basic; VIP $9.99/mo"},
{"name":"Tandem AI","category":"Language Exchange","description":"language exchange partner practice video text voice native correction","url":"https://tandem.net","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free matching; Pro $13.99/mo"},
{"name":"italki AI","category":"Language Tutoring","description":"language tutor online lesson teacher 1on1 practice marketplace","url":"https://italki.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free community; tutors $5-$30/hr"},
{"name":"Google Translate AI","category":"Translation","description":"translate translation language text free multilingual 130 languages","url":"https://translate.google.com","pricing_model":"Free","rating":"4.7/5","pricing_details":"100% free; API 500K chars free"},
{"name":"Microsoft Translator AI","category":"Translation","description":"translate translation language text document multilingual real-time","url":"https://translator.microsoft.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free apps; API 2M chars free/mo"},
{"name":"Amazon Translate AI","category":"Translation","description":"translate translation language text batch api enterprise neural aws","url":"https://aws.amazon.com/translate","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"2M chars/mo free 12mo; $15/M chars"},
{"name":"Unbabel AI","category":"Translation Service","description":"translation service human quality enterprise multilingual customer support","url":"https://unbabel.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise per volume"},
{"name":"Lilt AI","category":"Enterprise Translation","description":"translation adaptive enterprise localization cat tool context-aware human","url":"https://lilt.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise per volume"},
{"name":"Smartling AI","category":"Localization","description":"translation localization enterprise automation workflow visual tms","url":"https://smartling.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise per content volume"},
{"name":"Crowdin AI","category":"Localization","description":"localization translation crowdsource software app multilingual developer","url":"https://crowdin.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free OSS; Pro $50/mo"},
{"name":"Lokalise AI","category":"Localization","description":"localization translation app software multilingual workflow continuous","url":"https://lokalise.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"14-day trial; Essential $120/mo"},
{"name":"Phrase AI","category":"Localization","description":"localization translation software workflow enterprise cat generative ai","url":"https://phrase.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"14-day trial; ~$27/user/mo"},
{"name":"Otter.ai","category":"Transcription","description":"transcription meeting notes audio record live caption action items","url":"https://otter.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"300 min/mo; Pro $16.99/mo"},
{"name":"Rev AI","category":"Transcription API","description":"transcription subtitle caption audio video service api developer","url":"https://rev.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"5-hour trial; $0.02/min auto; $1.50/min human"},
{"name":"Sonix AI","category":"Transcription","description":"transcription audio video translate subtitle caption interactive editor","url":"https://sonix.ai","pricing_model":"Paid","rating":"4.8/5","pricing_details":"30 min trial; $10/hour"},
{"name":"Trint AI","category":"Transcription","description":"transcription audio video interview edit text search journalist","url":"https://trint.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"7-day trial; Starter $52/user/mo"},
{"name":"Happy Scribe AI","category":"Subtitling","description":"transcription subtitle caption video audio translate proofreading","url":"https://happyscribe.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free trial; Basic $10/mo"},
{"name":"Temi AI","category":"Transcription","description":"transcription audio video speech to text fast cheap timestamp","url":"https://temi.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"First 45 min free; $0.25/min"},
{"name":"TurboScribe AI","category":"Transcription","description":"transcription audio video unlimited accurate fast whisper 90 languages","url":"https://turboscribe.ai","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"3 files/day free; Pro $20/mo"},
{"name":"Sembly AI","category":"Meeting Assistant","description":"meeting transcription notes summary action items sync project management","url":"https://sembly.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"4 hours/mo; Professional $15/mo"},
{"name":"Fathom AI","category":"Meeting Assistant","description":"meeting transcription notes summary zoom record free highlight","url":"https://fathom.video","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free personal; Teams $19/user/mo"},
{"name":"Tactiq AI","category":"Meeting Transcription","description":"meeting transcription notes caption zoom google meet teams speaker","url":"https://tactiq.io","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"10 meetings/mo; Pro $12/mo"},
{"name":"Notta AI","category":"Transcription","description":"transcription audio video meeting notes summary translate 58 languages","url":"https://notta.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"120 min/mo; Pro $13.99/mo"},
{"name":"Airgram AI","category":"Meeting Assistant","description":"meeting recording transcription notes agenda action clip share","url":"https://airgram.io","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"5 recordings/mo; Plus $18/user/mo"},
{"name":"Subly AI","category":"Subtitling","description":"subtitle caption video translate audio accessibility multilingual","url":"https://subly.app","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; Business $19/mo"},
{"name":"Climate FieldView AI","category":"Agriculture","description":"agriculture farming field data crop yield monitor satellite digital","url":"https://climate.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"~$299/yr per farm"},
{"name":"CropX AI","category":"Soil & Irrigation","description":"soil sensor agriculture farming irrigation sensor data moisture","url":"https://cropx.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Hardware + software per acre"},
{"name":"Taranis AI","category":"Crop Monitoring","description":"agriculture crop monitoring leaf disease aerial scout precision imagery","url":"https://taranis.ag","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Per acreage scouted"},
{"name":"Prospera AI","category":"Greenhouse Farming","description":"agriculture greenhouse crop monitoring disease detection irrigation valmont","url":"https://prospera.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Per pivot/acreage"},
{"name":"Descartes Labs AI","category":"Geospatial Intelligence","description":"satellite agriculture earth observation crop predict yield geospatial","url":"https://descarteslabs.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise subscription"},
{"name":"Granular AI","category":"Farm Management","description":"farm management agriculture operation crop planning profitability corteva","url":"https://granular.ag","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Per-acre enterprise"},
{"name":"Pachama AI","category":"Carbon Credits","description":"carbon credit forest monitoring satellite nature offset reforestation","url":"https://pachama.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per credit volume"},
{"name":"Watershed AI","category":"Carbon Accounting","description":"carbon footprint sustainability emission enterprise report scope 1 2 3","url":"https://watershed.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"~$50,000/yr enterprise"},
{"name":"Persefoni AI","category":"Carbon Accounting","description":"carbon footprint accounting sustainability emission esg disclosure","url":"https://persefoni.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free single user; Enterprise custom"},
{"name":"Sweep AI","category":"Sustainability Platform","description":"carbon emission sustainability net zero climate enterprise supply chain","url":"https://sweep.net","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise per org size"},
{"name":"Siemens MindSphere AI","category":"Industrial IoT","description":"industrial iot manufacturing data machine analytics insights hub","url":"https://siemens.com/mindsphere","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Usage-based enterprise"},
{"name":"GE Predix AI","category":"Industrial IoT","description":"industrial iot manufacturing machine data analytics predix apm","url":"https://ge.com/digital/predix","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise contract"},
{"name":"PTC ThingWorx AI","category":"Industrial IoT","description":"industrial iot manufacturing smart connected factory ar augmented reality","url":"https://ptc.com/en/products/iot","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise license"},
{"name":"SAP Digital Manufacturing AI","category":"Manufacturing","description":"manufacturing production planning factory enterprise erp mes cloud","url":"https://sap.com/products/digital-manufacturing","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise user/facility"},
{"name":"AVEVA AI","category":"Industrial Software","description":"industrial software manufacturing engineering operations digital twin scada","url":"https://aveva.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Flex credits enterprise"},
{"name":"Seeq AI","category":"Process Analytics","description":"manufacturing process data analytics time series industrial yield quality","url":"https://seeq.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per data scale"},
{"name":"Cognex AI","category":"Machine Vision","description":"machine vision inspection manufacturing quality defect barcode deep learning","url":"https://cognex.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Hardware + software"},
{"name":"Blue Yonder AI","category":"Supply Chain","description":"supply chain management demand forecast inventory logistics optimization","url":"https://blueyonder.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise custom"},
{"name":"Kinaxis AI","category":"Supply Chain","description":"supply chain management concurrent planning logistics forecast rapidresponse","url":"https://kinaxis.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Enterprise custom"},
{"name":"Coupa AI","category":"Spend Management","description":"business spend management procurement invoice supply chain sourcing","url":"https://coupa.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Per annual spend"},
{"name":"Llamasoft AI","category":"Supply Chain Design","description":"supply chain design network optimization simulation logistics modeling","url":"https://coupa.com/products/supply-chain","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise license"},
{"name":"Manhattan Associates AI","category":"Warehouse Management","description":"supply chain warehouse wms omni commerce logistics transportation","url":"https://manh.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Enterprise contract"},
{"name":"NewsGPT AI","category":"News Generation","description":"news article generation journalism report auto write algorithmic","url":"https://newsgpt.ai","pricing_model":"Free","rating":"4.2/5","pricing_details":"Free access"},
{"name":"Narrative Science AI","category":"Data Narration","description":"natural language generation data to text report auto narrative tableau","url":"https://narrativescience.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Bundled with Tableau"},
{"name":"Automated Insights AI","category":"Data Narration","description":"natural language generation data report content auto wordsmith","url":"https://automatedinsights.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Per volume generated"},
{"name":"Arria AI","category":"Data Narration","description":"natural language generation data to text report content nlg engine","url":"https://arria.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Per API usage"},
{"name":"NewsWhip AI","category":"Media Monitoring","description":"news prediction trending story social media journalist viral velocity","url":"https://newswhip.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"14-day trial; custom"},
{"name":"Muck Rack AI","category":"PR & Media","description":"journalist database media contact press find pr pitch tracking","url":"https://muckrack.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Annual subscription"},
{"name":"Prowly AI","category":"PR & Media","description":"pr media contact press release journalist pitch ai generator brand","url":"https://prowly.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"7-day trial; Essential $258/mo"},
{"name":"Cision AI","category":"PR & Media Intelligence","description":"pr media monitoring press release communication distribution intelligence","url":"https://cision.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise contract"},
{"name":"Eventbrite AI","category":"Event Management","description":"event planning ticket create manage registration discover free paid","url":"https://eventbrite.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free for free events; fees per ticket"},
{"name":"Cvent AI","category":"Event Management","description":"event management venue conference registration corporate hybrid","url":"https://cvent.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Enterprise per modules"},
{"name":"Bizzabo AI","category":"Event Management","description":"event management conference networking registration hybrid experience","url":"https://bizzabo.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Annual enterprise"},
{"name":"Whova AI","category":"Event App","description":"event app conference attendee networking agenda engagement","url":"https://whova.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per event attendee count"},
{"name":"Hopin AI","category":"Virtual Events","description":"virtual event conference online networking webinar stage breakout","url":"https://hopin.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; Business custom"},
{"name":"Airmeet AI","category":"Virtual Events","description":"virtual event webinar conference networking online expo interactive","url":"https://airmeet.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free webinar; Premium $199/mo"},
{"name":"Splash AI","category":"Event Marketing","description":"event marketing invitation rsvp guest list manage landing page","url":"https://splashthat.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"Free individual; Enterprise ~$12,500/yr"},
{"name":"vFairs AI","category":"Virtual Events","description":"virtual event trade show expo conference 3d venue booth streaming","url":"https://vfairs.com","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Per event scope"},
{"name":"Vetster AI","category":"Pet Telehealth","description":"pet veterinary online telehealth vet consultation animal video","url":"https://vetster.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free registration; ~$50-$70/visit"},
{"name":"Petriage AI","category":"Pet Health","description":"pet health symptom assessment veterinary triage animal urgency","url":"https://petriage.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free basic; vet partnership"},
{"name":"WhiskerDocs AI","category":"Pet Telehealth","description":"pet veterinary telehealth consult advice animal 24/7 health","url":"https://whiskerdocs.com","pricing_model":"Paid","rating":"4.6/5","pricing_details":"~$40 consult; $16.99/mo"},
{"name":"PetsApp AI","category":"Vet Practice","description":"veterinary clinic pet care communication client app booking","url":"https://petsapp.com","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Per practice"},
{"name":"Anivive AI","category":"Animal Health","description":"pet pharmaceutical veterinary medicine drug discovery animal","url":"https://anivive.com","pricing_model":"Paid","rating":"4.5/5","pricing_details":"B2B pharma licensing"},
{"name":"FamilyWall AI","category":"Family Organizer","description":"family organizer calendar share schedule grocery list meal location","url":"https://familywall.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Premium $4.99/mo"},
{"name":"Cozi AI","category":"Family Organizer","description":"family organizer calendar schedule list meal planning grocery color-coded","url":"https://cozi.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free ad-supported; Gold $29.99/yr"},
{"name":"Bark AI","category":"Parental Control","description":"parental control monitor child phone screen safety text social media","url":"https://bark.us","pricing_model":"Paid","rating":"4.5/5","pricing_details":"Bark Jr $5/mo; Premium $14/mo"},
{"name":"Qustodio AI","category":"Parental Control","description":"parental control screen time monitor child safety web filter app block","url":"https://qustodio.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free 1 device; Premium ~$54.95/yr"},
{"name":"Life360 AI","category":"Family Safety","description":"family location tracking safety gps share driving crash detection","url":"https://life360.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free tracking; Silver/Gold $4.99-$19.99/mo"},
{"name":"Hallow AI","category":"Religion & Spirituality","description":"prayer meditation catholic religion spiritual faith audio","url":"https://hallow.com","pricing_model":"Freemium","rating":"4.9/5","pricing_details":"Free core; Plus $9.99/mo"},
{"name":"Pray.com AI","category":"Religion & Spirituality","description":"christian daily devotionals bedtime bible stories audio prayer","url":"https://pray.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free daily; Premium $11.99/mo"},
{"name":"Abide AI","category":"Religion & Spirituality","description":"christian meditation prayer bible sleep spiritual","url":"https://abide.co","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free daily; Premium $7.99/mo"},
{"name":"Glorify AI","category":"Religion & Spirituality","description":"christian devotion prayer bible daily worship","url":"https://glorify-app.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free core; Plus $8.99/mo"},
{"name":"Soultime AI","category":"Religion & Spirituality","description":"christian mindfulness app focusing on spiritual wellness emotional health prayer sessions and meditation music","url":"https://soultime.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free starter; $9.99/mo"},
{"name":"Deep Dream Generator","category":"Art & Creativity","description":"ai art generator neural style transfer deep learning digital art","url":"https://deepdreamgenerator.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free energy; $19/mo"},
{"name":"StarryAI","category":"Art & Creativity","description":"ai art generator app text to image custom illustration anime 3d","url":"https://starryai.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"5 credits/day; Pro $11.99/mo"},
{"name":"Wonder AI","category":"Art & Creativity","description":"mobile ai text-to-image generator artwork avatars paintings","url":"https://wonder-ai.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free ads; Pro $4.99/wk"},
{"name":"Dream by WOMBO","category":"Art & Creativity","description":"ai digital art platform painting artwork text prompt artistic","url":"https://wombo.art","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free basic; Premium $9.99/mo"},
{"name":"Craiyon","category":"Art & Creativity","description":"free web text-to-image generator drawing illustration prompt","url":"https://craiyon.com","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"100% free with ads; $5/mo supporter"},
{"name":"Autodraw","category":"Art & Creativity","description":"google ai experiment sketch drawing doodle icon recognition assist","url":"https://autodraw.com","pricing_model":"Free","rating":"4.6/5","pricing_details":"100% free web tool"},
{"name":"Pollinations AI","category":"Art & Creativity","description":"open-source generative ai platform free text-to-image api digital media","url":"https://pollinations.ai","pricing_model":"Free","rating":"4.5/5","pricing_details":"100% free open api"},
{"name":"Flaticon AI","category":"Icon Generation","description":"vector graphics marketplace icon generator text to vector symbol ui","url":"https://flaticon.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free with attribution; Premium $12.99/mo"},
{"name":"Depositphotos AI","category":"Stock Generation","description":"stock content licensing library ai image generator royalty free commercial","url":"https://depositphotos.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free trial; $29/mo"}
]'''

# NEW: additional tools appended for the August 2026 dataset expansion (kept
# separate from the original DATASET_JSON block above so the huge original
# string never has to be hand-edited). Merged into TOOLS below.
DATASET_JSON_ADDITIONS = r'''[
{"name":"Grok","category":"General AI","description":"xai assistant real time x data reasoning image generation deepsearch","url":"https://grok.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; SuperGrok $30/mo"},
{"name":"GLM","category":"General AI","description":"zhipu ai chat assistant reasoning coding open source long context","url":"https://chat.z.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free web chat; Coding Plan from $10/mo"},
{"name":"Qwen Chat","category":"General AI","description":"alibaba assistant reasoning coding multilingual open weight","url":"https://chat.qwen.ai","pricing_model":"Free","rating":"4.6/5","pricing_details":"Free web chat; API pay-per-token"},
{"name":"Mistral Le Chat","category":"General AI","description":"european assistant reasoning coding image generation web search sovereign","url":"https://chat.mistral.ai","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; Pro $14.99/mo"},
{"name":"Kimi","category":"General AI","description":"moonshot ai assistant long context reasoning agent deep research","url":"https://kimi.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free chat tier; paid plans from $19/mo"},
{"name":"Meta AI","category":"General AI","description":"meta llama assistant chat image generation instagram whatsapp integration","url":"https://meta.ai","pricing_model":"Free","rating":"4.3/5","pricing_details":"Completely free"},
{"name":"HuggingChat","category":"General AI","description":"open source chat interface multiple open weight models free","url":"https://huggingface.co/chat","pricing_model":"Free","rating":"4.4/5","pricing_details":"100% free open source"},
{"name":"Cohere","category":"General AI","description":"enterprise llm platform command model reasoning rag retrieval business","url":"https://cohere.com","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free trial API; pay-per-token"},
{"name":"Manus","category":"AI Agents","description":"autonomous ai agent research browse code slides virtual computer","url":"https://manus.im","pricing_model":"Freemium","rating":"4.5/5","pricing_details":"Free tier; Pro from $20/mo"},
{"name":"Genspark","category":"AI Agents","description":"super agent research slides spreadsheets phone calls video generation","url":"https://genspark.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free tier; Plus $24.99/mo"},
{"name":"Claude Code","category":"AI Code Assistants","description":"anthropic terminal coding agent codebase refactor multi-file autonomous","url":"https://claude.com/product/claude-code","pricing_model":"Paid","rating":"4.8/5","pricing_details":"Requires Claude Pro $20/mo+; API pay-per-token"},
{"name":"Devin","category":"Autonomous Coding Agents","description":"cognition ai autonomous software engineer plan code test pull request","url":"https://devin.ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free tier; Core from $20/mo"},
{"name":"Windsurf","category":"AI Code Editors","description":"ai native code editor cascade agent multi-file codebase understanding","url":"https://windsurf.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; Pro $20/mo"},
{"name":"Cline","category":"AI Code Assistants","description":"open source autonomous coding agent vscode extension terminal bring your own key","url":"https://cline.bot","pricing_model":"Free","rating":"4.7/5","pricing_details":"Free OSS (bring your own API key)"},
{"name":"JetBrains AI","category":"AI Code Assistants","description":"jetbrains ide assistant code completion chat refactor intellij pycharm","url":"https://jetbrains.com/ai","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free limited; AI Pro ~$10/mo"},
{"name":"Warp","category":"AI Terminal","description":"ai powered terminal command line agent workflow blocks autocomplete","url":"https://warp.dev","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; Pro ~$15/mo"},
{"name":"Bolt.new","category":"AI App Building","description":"stackblitz browser full stack app builder react webcontainers deploy","url":"https://bolt.new","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free 1M tokens/mo; Pro $25/mo"},
{"name":"Lovable","category":"AI App Building","description":"ai full stack app builder react supabase authentication deploy prompt","url":"https://lovable.dev","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; Pro $25/mo"},
{"name":"Flux","category":"Image Generation","description":"black forest labs image generation photorealistic open weight text to image","url":"https://blackforestlabs.ai","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free via partners; API pay-per-image"},
{"name":"Google Whisk","category":"Image Generation","description":"google image generation remix image prompts imagen model creative","url":"https://labs.google/whisk","pricing_model":"Free","rating":"4.5/5","pricing_details":"Free with Google account"},
{"name":"Grok Imagine","category":"Image Generation","description":"xai image and short video generation text to image spicy mode","url":"https://grok.com","pricing_model":"Paid","rating":"4.3/5","pricing_details":"Included in SuperGrok $30/mo"},
{"name":"Kling AI","category":"Video Generation","description":"kuaishou text to video image to video native audio motion control cinematic","url":"https://klingai.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; Standard ~$6.99/mo"},
{"name":"Hailuo AI","category":"Video Generation","description":"minimax text to video image to video fast generation physics realism","url":"https://hailuoai.video","pricing_model":"Freemium","rating":"4.4/5","pricing_details":"Free tier; from ~$9.99/mo"},
{"name":"Google Veo","category":"Video Generation","description":"google deepmind cinematic video generation text to video audio native","url":"https://deepmind.google/technologies/veo","pricing_model":"Paid","rating":"4.7/5","pricing_details":"Included in Google AI Pro/Ultra plans"},
{"name":"Vidu","category":"Video Generation","description":"shengshu ai text to video image to video character consistency anime","url":"https://vidu.com","pricing_model":"Freemium","rating":"4.3/5","pricing_details":"Free daily credits; paid plans from ~$10/mo"},
{"name":"CapCut","category":"Video Editing","description":"bytedance video editor auto caption template ai effects social media","url":"https://capcut.com","pricing_model":"Freemium","rating":"4.7/5","pricing_details":"Free; Pro ~$9.99/mo"},
{"name":"NotebookLM","category":"Research & Academia","description":"google source grounded research notes audio overview podcast citations","url":"https://notebooklm.google.com","pricing_model":"Freemium","rating":"4.8/5","pricing_details":"Free with Google account; Plus bundle from ~$4.99/mo"},
{"name":"Microsoft 365 Copilot","category":"Business Intelligence","description":"microsoft office assistant word excel powerpoint outlook teams workplace","url":"https://microsoft.com/microsoft-365/copilot","pricing_model":"Paid Add-on","rating":"4.4/5","pricing_details":"$30/user/mo add-on"},
{"name":"Zapier AI","category":"Workflow Automation","description":"workflow automation connect apps ai agent trigger action no-code","url":"https://zapier.com","pricing_model":"Freemium","rating":"4.6/5","pricing_details":"Free tier; Starter ~$19.99/mo"},
{"name":"Adobe Photoshop","category":"Photo Editing","description":"generative fill remove object expand image ai photo edit industry standard","url":"https://adobe.com/products/photoshop","pricing_model":"Paid","rating":"4.7/5","pricing_details":"~$22.99/mo (Photography plan)"}
]'''

# Parse JSON and create deduplicated list
_RAW_TOOLS = json.loads(DATASET_JSON) + json.loads(DATASET_JSON_ADDITIONS)
_seen_names = set()
_deduped_tools = []
for t in _RAW_TOOLS:
    if t['name'] not in _seen_names:
        _seen_names.add(t['name'])
        _deduped_tools.append(t)

# Enrich tools with inferred data
def _slugify(name): return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
def _enrich(tool):
    is_general = tool["category"] == "General AI"
    desc = tool.get("description", "").lower()
    pricing = tool.get("pricing_model", "").lower()
    details = tool.get("pricing_details", "").lower()

    is_free = "free" in pricing or "free" in details
    is_open_source = "open source" in desc or "open source" in details
    has_api = "api" in desc or "api" in details

    return {
        **tool,
        "id": _slugify(tool["name"]),
        "popularity": float(tool.get("rating", "4.0/5").split("/")[0]) / 5.0,
        "verified": TODAY,
        "specialized": not is_general,
        "best_for": "General tasks" if is_general else tool["category"],
        "features": [tool["category"], tool["pricing_model"], "AI-powered"],
        "is_free": is_free,
        "is_open_source": is_open_source,
        "has_api": has_api
    }

TOOLS = [_enrich(t) for t in _deduped_tools]
# ============================================================
# AI Tool Recommender - EXACT MATCH & LOGIC (Part 2)
# ============================================================
print(f"Loaded {len(TOOLS)} tools. Loading model...")
_sbert = SentenceTransformer("all-MiniLM-L6-v2")
_descriptions = [t["description"] for t in TOOLS]
_embeddings = _sbert.encode(_descriptions, normalize_embeddings=True)
_tfidf = TfidfVectorizer(stop_words="english")
_tfidf_matrix = _tfidf.fit_transform(_descriptions)
print("Done.")

def embed_query(text): return _sbert.encode([text], normalize_embeddings=True)[0]
def embed_batch(texts): return _sbert.encode(texts, normalize_embeddings=True) if texts else np.array([])

# ---------------------------------------------------------------
# NAME MATCHING HELPERS (used by recommend() for exact/near-exact
# tool-name searches like "Claude" or "GLM")
# ---------------------------------------------------------------
_NAME_MATCH_STOPWORDS = {"ai", "app", "the", "io"}

def _name_match_tokens(text):
    """Lowercase, strip punctuation, split into words, and drop generic
    filler words like 'ai'/'app' -- but only if that doesn't empty the
    list (so a tool literally named 'AI' still matches)."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [tok for tok in cleaned.split() if tok]
    filtered = [tok for tok in tokens if tok not in _NAME_MATCH_STOPWORDS]
    return filtered if filtered else tokens

def find_name_matches(query):
    """Return tools whose name strongly matches the query as a NAME
    (not a task description), so a search for 'Claude', 'claude ai',
    or 'GLM' surfaces only that tool instead of blending in unrelated
    semantic matches. Returns [] when the query doesn't look like a
    tool-name search at all."""
    q_tokens = _name_match_tokens(query)
    if not q_tokens:
        return []
    q_norm = " ".join(q_tokens)
    # Guard against very short/generic queries (e.g. a lone "ai") matching
    # nearly everything.
    if len(q_norm) < 3:
        return []

    exact, contained = [], []
    for t in TOOLS:
        name_tokens = _name_match_tokens(t["name"])
        name_norm = " ".join(name_tokens)
        if name_norm == q_norm:
            exact.append(t)
        elif (name_norm.startswith(q_norm + " ") or q_norm.startswith(name_norm + " ")
              or (len(q_tokens) == 1 and q_tokens[0] in name_tokens)):
            contained.append(t)

    return exact if exact else contained

# ---------------------------------------------------------------
# FEEDBACK & DETERMINISTIC SCORING LOGIC
# ---------------------------------------------------------------
FEEDBACK_FILE = "feedback.json"
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_feedback(query, tool_id):
    fb = load_feedback()
    fb.append({"query": query, "tool_id": tool_id, "ts": datetime.utcnow().isoformat()})
    with open(FEEDBACK_FILE, "w") as f: json.dump(fb, f, indent=2)

def recommend(query, top_k=5, specialized_only=False):
    if not query or not query.strip(): return []

    cat_filter = None
    search_query = query
    if query.startswith("category:"):
        cat_filter = query.split(":", 1)[1].strip()
        search_query = cat_filter

    q_lower = search_query.lower().strip()

    # NAME MATCH: if the query looks like a tool name ("Claude", "claude ai",
    # "GLM"...) rather than a task description, return ONLY the matching
    # tool(s) immediately instead of blending in semantic matches.
    if not cat_filter:
        name_matches = find_name_matches(search_query)
        if name_matches:
            return [{**t, "score": 99} for t in name_matches][:top_k]
    else:
        # Preserve the previous exact-match behaviour for category filters
        exact_matches = [t for t in TOOLS if t["name"].lower().strip() == q_lower]
        if exact_matches:
            return [{**t, "score": 99} for t in exact_matches][:top_k]

    q_vec = embed_query(search_query)
    sem_scores = _embeddings @ q_vec
    kw_scores = cosine_similarity(_tfidf.transform([search_query]), _tfidf_matrix)[0]

    fb = load_feedback()
    fb_boost = np.zeros(len(TOOLS))
    if fb:
        past_vecs = embed_batch([f["query"] for f in fb])
        sims = past_vecs @ q_vec
        for f, sim in zip(fb, sims):
            tid = f["tool_id"]
            for i, t in enumerate(TOOLS):
                if t["id"] == tid:
                    fb_boost[i] = max(fb_boost[i], float(sim))
                    break

    results = []
    for i, tool in enumerate(TOOLS):
        if cat_filter and tool["category"] != cat_filter:
            continue

        if specialized_only and not tool.get("specialized", False) and not cat_filter:
            continue

        task_rel = float(sem_scores[i]) * 0.40
        cat_rel = float(kw_scores[i]) * 0.25
        feat_rel = float(sem_scores[i]) * 0.20
        spec_rel = 0.10 if tool.get("specialized", False) else 0.0
        other = float(fb_boost[i]) * 0.05

        raw_score = task_rel + cat_rel + feat_rel + spec_rel + other
        score = min(99, max(35, int(raw_score * 100 + 20)))

        results.append({**tool, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)

    if cat_filter:
        return results[:50]

    return results[:top_k]

# ---------------------------------------------------------------
# "WHY THIS MATCH?" GENERATOR
# ---------------------------------------------------------------
FILLER_WORDS = {"best", "top", "ai", "general", "purpose", "assistant", "tool", "software"}
def generate_why_reasons(tool, query):
    reasons = []
    desc = tool.get("description", "").lower()
    cat = tool.get("category", "")
    pricing = tool.get("pricing_model", "")

    if tool.get("specialized", False):
        reasons.append(f"✓ Specialized in {cat}")
    else:
        reasons.append("✓ General purpose AI assistant")

    q_words = [w for w in query.lower().split() if len(w) > 3 and w not in FILLER_WORDS]
    matched_kw = [w for w in q_words if w in desc]
    if matched_kw:
        kw_str = matched_kw[0]
        if len(matched_kw) > 1: kw_str += f" and {matched_kw[1]}"
        reasons.append(f"✓ Great for {kw_str} tasks")
    else:
        reasons.append(f"✓ Designed for {tool.get('best_for', 'various tasks')}")

    if "api" in desc: reasons.append("✓ API available for developers")
    elif "free" in pricing.lower(): reasons.append("✓ Free tier available")
    elif "open source" in desc: reasons.append("✓ Open source platform")
    else: reasons.append(f"✓ {pricing} pricing model")

    return '<br>'.join(reasons)

# ---------------------------------------------------------------
# COMPARISON LOGIC & STRICT 5-TOOL LIMIT
# ---------------------------------------------------------------
def enforce_compare_limit(selected_tools):
    if len(selected_tools) > 5:
        gr.Warning("You can compare up to 5 tools at a time.")
        return selected_tools[:5]
    return selected_tools

def compare_tools(selected_tools, current_query):
    if not selected_tools:
        return "<div style='color:#F4B72A; text-align:center; padding:20px;'>Please select 1 to 5 tools to compare.</div>"

    if len(selected_tools) > 5:
        selected_tools = selected_tools[:5]

    tools_to_compare = [t for t in TOOLS if t["name"] in selected_tools][:5]

    query = current_query.strip() if current_query and current_query.strip() else "best ai"
    q_vec = embed_query(query)
    sem_scores = _embeddings @ q_vec
    kw_scores = cosine_similarity(_tfidf.transform([query]), _tfidf_matrix)[0]

    tool_scores = {}
    for i, tool in enumerate(TOOLS):
        task_rel = float(sem_scores[i]) * 0.40
        cat_rel = float(kw_scores[i]) * 0.25
        feat_rel = float(sem_scores[i]) * 0.20
        spec_rel = 0.10 if tool.get("specialized", False) else 0.0
        raw_score = task_rel + cat_rel + feat_rel + spec_rel
        score = min(99, max(35, int(raw_score * 100 + 20)))
        if query.lower().strip() == tool["name"].lower().strip():
            score = 99
        tool_scores[tool["name"]] = score

    criteria = ["Match Score", "Rating", "Pricing Model", "Free Tier?", "Open Source?", "API Available?"]

    html = '<div class="results-header"><span class="results-title">Tool Comparison</span></div>'
    html += '<table class="compare-table"><thead><tr><th>Feature</th>'
    for t in tools_to_compare: html += f'<th>{t["name"]}</th>'
    html += '</tr></thead><tbody>'

    for crit in criteria:
        html += f'<tr><td>{crit}</td>'
        for t in tools_to_compare:
            val = "-"
            if crit == "Match Score": val = f'{tool_scores.get(t["name"], 0)}%'
            elif crit == "Rating": val = f'★ {t.get("rating", "N/A")}'
            elif crit == "Pricing Model": val = t.get("pricing_model", "N/A")
            elif crit == "Free Tier?": val = "✓" if t.get("is_free") else "✗"
            elif crit == "Open Source?": val = "✓" if t.get("is_open_source") else "✗"
            elif crit == "API Available?": val = "✓" if t.get("has_api") else "✗"

            if val == "✓": val = '<span style="color:#34D978; font-weight:bold;">✓</span>'
            elif val == "✗": val = '<span style="color:#F87171; font-weight:bold;">✗</span>'

            html += f'<td>{val}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html

def feedback_handler(task, choice):
    if not choice: return "Pick a tool from the list above first."
    tool = next((t for t in TOOLS if t["name"] == choice), None)
    if not tool: return "Something went wrong, couldn't find that tool."
    save_feedback(task, tool["id"])
    return f"Thanks - noted that **{choice}** worked for you."
# ============================================================
# AI Tool Recommender - UI POLISH & RELOAD STATE (Part 3)
# ============================================================
def _clean_words(desc): return [w for w in desc.split() if w.lower() not in FILLER_WORDS]
def _format_description(desc, name):
    words = _clean_words(desc)
    if not words: return f"{name} is a versatile AI-powered tool."
    if len(words) == 1: return f"Focused on {words[0]}."
    return f"Helps with {', '.join(words[:-1])} and {words[-1]}."
def _format_tags(desc, max_tags=3):
    words, tags, seen = _clean_words(desc), [], set()
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl); tags.append(w.capitalize())
            if len(tags) >= max_tags: break
    return tags
def _get_logo_url(url):
    try:
        domain = urlparse(url).netloc
        return f"https://www.google.com/s2/favicons?sz=128&domain_url=https://{domain}" if domain else ""
    except: return ""

def build_card(t, query=""):
    logo_url = _get_logo_url(t["url"])
    desc_text = _format_description(t["description"], t["name"])
    tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in _format_tags(t["description"])])
    score = int(t.get("score", 0))
    tool_id = t.get("id", "tool")

    why_reasons = generate_why_reasons(t, query)

    why_html = f"""
    <div class="why-container">
      <input type="checkbox" class="why-checkbox" id="why-{tool_id}">
      <label for="why-{tool_id}" class="why-trigger">Why this match?</label>
      <div class="why-content">{why_reasons}</div>
    </div>
    """

    return f"""
    <a href="{t['url']}" target="_blank" class="result-card">
        <div class="card-inner">
            <div class="card-left">
                <img src="{logo_url}" class="tool-logo" alt="{t['name']}">
                <div class="card-main">
                    <div class="tool-header">
                        <span class="tool-name">{t['name']}</span>
                        <span class="tool-category">{t.get('category', '')}</span>
                    </div>
                    <div class="tool-desc">{desc_text}</div>
                    <div class="tool-meta">
                        <span class="meta-rating">★ {t.get('rating', 'N/A')}</span>
                        <span class="meta-pricing">{t.get('pricing_model', 'N/A')}</span>
                    </div>
                    <div class="tags-container">{tags_html}</div>
                    {why_html}
                </div>
            </div>
            <div class="card-match">
                <div class="ring-container" style="--p: {score}%;">
                    <div class="ring-bg"></div>
                    <div class="ring-inner"><span class="ring-text">{score}%</span></div>
                </div>
                <span class="view-btn">View →</span>
            </div>
        </div>
    </a>
    """

CSS = """
.gradio-container .block, .gradio-container .form, .gradio-container .gap { border: none !important; background: none !important; box-shadow: none !important; overflow: visible !important; }
.gradio-container { background: #0B0F0D !important; max-width: 1400px !important; padding: 0 !important; margin: 0 auto !important; font-family: 'Inter', sans-serif !important; min-height: 100vh !important; }
* { color: #F5F7F6 !important; }
.main-layout { gap: 0 !important; min-height: 100vh !important; }

.sidebar { background-color: #0D100E !important; padding: 24px 14px !important; border-right: 1px solid #26352D !important; min-height: 100vh; display: flex; flex-direction: column; }
.brand { display: flex; align-items: center; gap: 12px; padding: 0 4px 24px 4px; white-space: nowrap; }
.brand-icon { width: 36px; height: 36px; border-radius: 8px; border: 2px solid #22C55E; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0; background: #111714; }
.brand-text { font-size: 1.3rem; font-weight: 800; color: #F5F7F6 !important; letter-spacing: -0.5px; line-height: 1.1; }
.sidebar button { width: 100% !important; text-align: left !important; background: transparent !important; border: none !important; color: #AAB5AE !important; margin: 0 0 4px 0 !important; padding: 10px 12px !important; font-weight: 600 !important; font-size: 0.9rem !important; min-height: unset !important; border-radius: 6px !important; }
.sidebar button:hover { background: #111714 !important; color: #F5F7F6 !important; }
.sidebar button.nav-active { background: #111714 !important; color: #34D978 !important; border-left: 3px solid #34D978 !important; }
.sidebar-spacer { flex-grow: 1; }
.promo-card { background: #111714 !important; border: 1px solid #26352D !important; border-radius: 8px; padding: 12px !important; margin: 12px 2px !important; }
#promo-title { font-size: 0.85rem !important; font-weight: 700 !important; color: #34D978 !important; margin: 0 0 4px 0 !important; }
#promo-sub { font-size: 0.75rem !important; color: #AAB5AE !important; line-height: 1.3 !important; margin: 0 0 8px 0 !important; }
#promo-btn { background: transparent !important; border: none !important; color: #34D978 !important; font-size: 0.8rem !important; font-weight: 700 !important; padding: 0 !important; min-height: unset !important; text-align: left !important; }
#sidebar-footer { font-size: 0.65rem !important; color: #52525b !important; padding: 12px 4px 0 4px !important; }

.main-content { padding: 24px 28px !important; }
.header-row { margin-bottom: 20px !important; }
#header-title { font-size: 1.6rem !important; font-weight: 800 !important; color: #F5F7F6 !important; margin: 0 0 4px 0 !important; letter-spacing: -0.5px; }
#header-subtitle { color: #AAB5AE !important; font-size: 0.9rem !important; margin: 0 !important; line-height: 1.4; }

.search-shell { display: flex; align-items: stretch; background-color: #111714; border: 1px solid #26352D; border-radius: 8px; padding: 4px; margin-bottom: 8px; gap: 6px; }
.search-shell:focus-within { border-color: #34D978; }
#search-input textarea { background: transparent !important; border: none !important; padding: 8px 10px !important; font-size: 0.95rem !important; min-height: unset !important; box-shadow: none !important; }
#search-btn { background-color: #22C55E !important; border: none !important; border-radius: 6px !important; padding: 0 20px !important; font-weight: 700 !important; font-size: 0.9rem !important; min-height: unset !important; flex-shrink: 0; transition: all 0.2s ease; position: relative; overflow: hidden; }
#search-btn, #search-btn * { color: #052e16 !important; }
#search-btn:hover { background-color: #34D978 !important; }

/* SEARCH BUTTON RELOAD/LOADING STATE */
#search-btn:active { transform: scale(0.98); }
#search-btn:disabled,
#search-btn[disabled],
#search-btn.pending,
#search-btn[aria-disabled="true"] {
    background-color: #172019 !important;
    color: #34D978 !important;
    opacity: 0.8 !important;
    cursor: wait !important;
    border: 1px solid #34D978 !important;
}
/* Make sure the loading spinner is visible */
#search-btn .gradio-loading {
    border-top-color: #34D978 !important;
    border-right-color: #34D978 !important;
    border-bottom-color: transparent !important;
    border-left-color: transparent !important;
}

.specialized-toggle { padding: 4px 4px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
.specialized-toggle label { color: #AAB5AE !important; font-size: 0.85rem !important; cursor: pointer; }
.specialized-toggle input[type="checkbox"] { accent-color: #22C55E; cursor: pointer; }
.help-tip { display: inline-block; width: 14px; height: 14px; background: #26352D; color: #AAB5AE !important; border-radius: 50%; text-align: center; line-height: 14px; font-size: 0.65rem; cursor: help; position: relative; }
.help-tip:hover::after { content: attr(data-tip); position: absolute; left: 20px; top: -5px; background: #111714; border: 1px solid #26352D; padding: 6px 10px; border-radius: 4px; color: #F5F7F6 !important; font-size: 0.75rem; width: 180px; z-index: 10; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }

.results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; margin-top: 16px; }
.results-title { color: #F5F7F6 !important; font-size: 1.1rem !important; font-weight: 700 !important; }

a.result-card { display: block; background-color: #111714; border: 1px solid #26352D; border-radius: 10px; padding: 18px; margin-bottom: 12px; text-decoration: none !important; transition: all 0.2s; }
a.result-card:hover { background-color: #172019; border-color: #2A3A30; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.card-inner { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.card-left { display: flex; gap: 14px; flex-grow: 1; min-width: 0; }
.tool-logo { width: 40px; height: 40px; border-radius: 8px; object-fit: contain; background: #172019; padding: 4px; flex-shrink: 0; margin-top: 2px; }
.card-main { flex-grow: 1; min-width: 0; text-align: left; }
.tool-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; flex-wrap: wrap; }
.tool-name { font-size: 1.1rem; font-weight: 700; color: #F5F7F6 !important; }
.tool-category { font-size: 0.7rem; color: #AAB5AE !important; background: #0B0F0D; padding: 2px 6px; border-radius: 4px; border: 1px solid #26352D; }
.tool-desc { color: #AAB5AE !important; font-size: 0.85rem; line-height: 1.4; margin-bottom: 8px; }
.tool-meta { display: flex; gap: 12px; margin-bottom: 8px; font-size: 0.8rem; color: #AAB5AE !important; align-items: center; }
.meta-rating { color: #F4B72A !important; font-weight: 700; display: flex; align-items: center; gap: 4px; }
.meta-pricing { color: #34D978 !important; font-weight: 700; text-transform: capitalize; padding: 2px 6px; background: #0B0F0D; border: 1px solid #26352D; border-radius: 4px; font-size: 0.7rem; }
.tags-container { display: flex; gap: 6px; flex-wrap: wrap; }
.tag { background: #0B0F0D; border: 1px solid #26352D; color: #34D978 !important; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }

.why-container { margin-top: 10px; width: 100%; }
.why-checkbox { display: none; }
.why-trigger { cursor: pointer; color: #34D978; font-size: 0.75rem; text-decoration: underline; display: block; margin-bottom: 4px; }
.why-content { display: none; padding: 10px; background: #0B0F0D; border-radius: 6px; border: 1px solid #26352D; font-size: 0.8rem; color: #AAB5AE; line-height: 1.6; }
.why-checkbox:checked ~ .why-content { display: block; }

.card-match { display: flex; flex-direction: column; align-items: center; gap: 6px; flex-shrink: 0; width: 55px; margin-top: 4px; }
.ring-container { width: 42px; height: 42px; border-radius: 50%; position: relative; display: flex; align-items: center; justify-content: center; }
.ring-bg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-radius: 50%; background: conic-gradient(#34D978 var(--p), #26352D 0); }
.ring-inner { position: absolute; top: 3px; left: 3px; width: calc(100% - 6px); height: calc(100% - 6px); background: #111714; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.ring-text { font-size: 0.75rem; font-weight: 800; color: #F5F7F6; z-index: 1; }
.view-btn { color: #34D978 !important; font-size: 0.75rem; font-weight: 700; white-space: nowrap; }
a.result-card:hover .view-btn { color: #22C55E !important; }

.compare-group { background: #111714; border: 1px solid #26352D; border-radius: 10px; padding: 14px 16px; margin-top: 20px; }
.compare-group label { color: #AAB5AE !important; font-size: 0.85rem !important; }
#compare-btn { background: #F4B72A !important; border: none !important; font-weight: 700 !important; padding: 8px 20px !important; border-radius: 6px !important; font-size: 0.85rem !important; min-height: unset !important; color: #052e16 !important; }
#compare-btn, #compare-btn * { color: #052e16 !important; }
.compare-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.85rem; }
.compare-table th, .compare-table td { border: 1px solid #26352D; padding: 8px; text-align: left; color: #F5F7F6; }
.compare-table th { background: #172019; color: #34D978; font-weight: 700; }
.compare-table td:first-child { font-weight: bold; color: #AAB5AE; background: #0B0F0D; }

.feedback-card { background-color: #111714 !important; border: 1px solid #26352D !important; border-radius: 10px; padding: 14px 16px !important; margin-top: 16px; }
.fb-info { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.fb-icon-wrap { width: 32px; height: 32px; border-radius: 8px; background: #0B0F0D; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }
.fb-text-col { display: flex; flex-direction: column; gap: 2px; }
#fb-label-html { font-size: 0.9rem; font-weight: 700; color: #F5F7F6 !important; }
#fb-sub-html { font-size: 0.75rem; color: #AAB5AE !important; }
#fb-radio .wrap { display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 6px !important; margin-bottom: 8px !important; }
#fb-radio label { background: #0B0F0D !important; border: 1px solid #26352D !important; border-radius: 16px !important; padding: 4px 10px !important; font-size: 0.75rem !important; margin: 0 !important; display: inline-flex !important; align-items: center !important; cursor: pointer !important; transition: all 0.2s ease !important; color: #AAB5AE !important; }
#fb-radio label:hover { border-color: #2A3A30 !important; color: #F5F7F6 !important; }
#fb-radio label:has(input:checked) { border-color: #34D978 !important; background: #111714 !important; color: #34D978 !important; }
#fb-radio input[type="radio"] { display: none !important; }
.feedback-bottom { display: flex !important; justify-content: flex-end !important; align-items: center !important; gap: 12px !important; }
#fb-output { font-size: 0.8rem !important; color: #34D978 !important; margin: 0 !important; padding: 0 !important; }
#fb-submit-btn { background: #22C55E !important; border: none !important; font-weight: 700 !important; padding: 6px 16px !important; border-radius: 6px !important; font-size: 0.8rem !important; min-height: unset !important; width: auto !important; }
#fb-submit-btn, #fb-submit-btn * { color: #052e16 !important; }
#fb-submit-btn:hover { background: #34D978 !important; }

.right-sidebar { background-color: #0D100E !important; padding: 24px 16px !important; border-left: 1px solid #26352D !important; min-height: 100vh; }
.right-card { background: #111714; border: 1px solid #26352D; border-radius: 10px; padding: 16px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.right-title { font-size: 0.9rem; font-weight: 700; color: #F5F7F6; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 6px; }
.right-item { display: flex; align-items: center; gap: 10px; padding: 6px 0 !important; border-bottom: 1px solid #1A1F1C; transition: all 0.2s; text-decoration: none !important; }
.right-item:last-child { border-bottom: none; }
.right-item:hover { background: #0B0F0D; padding-left: 4px !important; border-radius: 4px; }
.right-item:hover .right-item-name { color: #34D978; }
.right-item-rank { font-size: 1rem; font-weight: 800; color: #F4B72A; width: 20px; text-align: center; font-family: monospace; text-shadow: 0 0 8px rgba(244, 183, 42, 0.3); }
.right-item-logo { width: 24px; height: 24px; border-radius: 4px; object-fit: contain; background: #172019; padding: 2px; flex-shrink: 0; }
.right-item-name { font-size: 0.8rem; color: #F5F7F6; font-weight: 600; transition: color 0.2s; text-align: left !important; }
.right-item-cat { font-size: 0.65rem; color: #AAB5AE; margin-top: 2px; text-align: left !important; }
.popular-task-icon { width: 24px; height: 24px; border-radius: 4px; background: #172019; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; color: #34D978; flex-shrink: 0; }

.cat-card { background: #111714; border: 1px solid #26352D; border-radius: 8px; padding: 16px; transition: all 0.2s; cursor: pointer; text-decoration: none !important; display: block; }
.cat-card:hover { border-color: #34D978; background: #172019; transform: translateY(-2px); }
.cat-name { color: #34D978; font-weight: 700; font-size: 1rem; margin-bottom: 6px; }
.cat-count { color: #AAB5AE; font-size: 0.8rem; }
"""

def search(task, specialized_only):
    query = task.strip() if task and task.strip() else "best ai"
    is_cat_search = query.startswith("category:")

    results = recommend(query, top_k=5, specialized_only=specialized_only)
    is_home = not task.strip()

    if not results or (results[0]["score"] < 25 and not is_cat_search):
        html = "<div style='text-align:center; color:#AAB5AE; padding: 40px; font-size: 1rem;'>No strong matches found. Try rephrasing what you're looking for.</div>"
    else:
        title = "Category Results" if is_cat_search else "Top Matches"
        html = f'<div class="results-header"><span class="results-title">{title}</span></div>'
        for t in results: html += build_card(t, query)

    choices = [t["name"] for t in results] if results else []
    radio_update = gr.update(choices=choices, visible=True if choices else False, value=None)
    compare_update = gr.update(choices=choices, visible=True if choices else False, value=None)

    if is_home:
        title_update = gr.update(value="# Find the Best AI Tools")
        sub_update = gr.update(value="Search, compare, and discover the right AI tools for what you're trying to accomplish.")
    elif is_cat_search:
        cat_name = query.split(":", 1)[1].strip()
        title_update = gr.update(value=f"# {cat_name} Tools")
        sub_update = gr.update(value=f"Showing all tools assigned to the {cat_name} category.")
    else:
        title_update = gr.update(value="# Search Results")
        sub_update = gr.update(value=f"Showing top matches for: **{query}**")

    btn_updates = [gr.update(elem_classes="nav-active"), gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes="")]

    # search_btn_update is a no-op update: it changes nothing about the
    # button, but including it in the return value/outputs list makes the
    # Search button join the same pending -> updated visual cycle as every
    # other component when switching sections (see view_outputs below).
    search_btn_update = gr.update()

    return html, radio_update, compare_update, title_update, sub_update, *btn_updates, query, search_btn_update

def show_home(specialized_only):
    res = search("", specialized_only)
    return res[:-2] + ("", gr.update())

def show_explore(specialized_only):
    filtered = [t for t in TOOLS if not specialized_only or t.get("specialized", False)]
    sample = random.sample(filtered, min(8, len(filtered)))
    html = '<div class="results-header"><span class="results-title">🧭 Explore AI Tools</span></div>'
    for t in sample: html += build_card(t, "explore")
    btn_updates = [gr.update(elem_classes=""), gr.update(elem_classes="nav-active"), gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes="")]
    choices = [t["name"] for t in sample]
    return html, gr.update(visible=False, value=None), gr.update(choices=choices, visible=True, value=None), gr.update(value="# Explore AI Tools"), gr.update(value="Discover new and exciting AI tools randomly selected from our database."), *btn_updates, "", gr.update()

def show_trending(specialized_only):
    results = recommend("best ai", top_k=5, specialized_only=specialized_only)
    html = '<div class="results-header"><span class="results-title">🔥 Trending AI Tools</span></div>'
    for t in results: html += build_card(t, "best ai")
    btn_updates = [gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes="nav-active"), gr.update(elem_classes=""), gr.update(elem_classes="")]
    choices = [t["name"] for t in results]
    return html, gr.update(visible=False, value=None), gr.update(choices=choices, visible=True, value=None), gr.update(value="# Trending AI Tools"), gr.update(value="The most popular and highly-rated AI tools right now."), *btn_updates, "", gr.update()

def show_categories():
    cats = Counter([t['category'] for t in TOOLS])
    html = '<div class="results-header"><span class="results-title">▦ Browse Categories</span></div>'
    html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">'
    for c, count in sorted(cats.items()):
        onclick_js = f"var ta = document.querySelector('#search-input textarea'); ta.value='category: {c}'; ta.dispatchEvent(new Event('input', {{bubbles:true}})); document.querySelector('#search-btn').click();"
        html += f'<a class="cat-card" onclick="{onclick_js}"><div class="cat-name">{c}</div><div class="cat-count">{count} Tools Available</div></a>'
    html += '</div>'
    btn_updates = [gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes="nav-active"), gr.update(elem_classes="")]
    return html, gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(value="# Browse Categories"), gr.update(value="Explore tools by their specific use cases and industries."), *btn_updates, "", gr.update()

def show_about():
    html = '<div style="text-align: center; padding: 40px 20px; background: #111714; border: 1px solid #26352D; border-radius: 16px; margin-top: 20px;"><div style="font-size: 2rem; margin-bottom: 12px;">✨</div><h2 style="color: #34D978; font-weight: 800; margin-bottom: 12px; font-size: 1.5rem;">Discover Better Tools</h2><p style="color: #AAB5AE; max-width: 500px; margin: 0 auto 20px auto; font-size: 1rem; line-height: 1.6;">AI Finder is your premium dashboard to search, compare, and discover the right AI tools for what you\'re trying to accomplish.</p><div style="display: inline-block; background: #0B0F0D; border: 1px solid #26352D; padding: 8px 16px; border-radius: 8px; color: #34D978; font-weight: 700; font-size: 0.9rem;">200+ Specialized AI Tools Indexed</div></div>'
    btn_updates = [gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes=""), gr.update(elem_classes="nav-active")]
    return html, gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(value="# About AI Finder"), gr.update(value="Learn more about our mission to index the best AI tools."), *btn_updates, "", gr.update()

with gr.Blocks(css=CSS) as app:
    with gr.Row(elem_classes="main-layout"):
        with gr.Column(scale=1, min_width=200, elem_classes="sidebar"):
            gr.HTML('<div class="brand"><span class="brand-icon">🔍</span><span class="brand-text">AI Finder</span></div>')
            home_btn = gr.Button("🏠 Home", variant="secondary", elem_classes="nav-active")
            explore_btn = gr.Button("🧭 Explore", variant="secondary")
            trending_btn = gr.Button("🔥 Trending", variant="secondary")
            categories_btn = gr.Button("▦ Categories", variant="secondary")
            gr.HTML('<div class="sidebar-spacer"></div>')
            with gr.Column(elem_classes="promo-card"):
                gr.Markdown("✨ Discover better tools", elem_id="promo-title")
                gr.Markdown("Specialist-first AI database.", elem_id="promo-sub")
                learn_more_btn = gr.Button("Learn More →", elem_id="promo-btn")
            gr.HTML('<div id="sidebar-footer">© 2025 AI Finder<br>All rights reserved.</div>')

        with gr.Column(scale=4, elem_classes="main-content"):
            with gr.Row(elem_classes="header-row"):
                with gr.Column():
                    title_md = gr.Markdown("# Find the Best AI Tools", elem_id="header-title")
                    subtitle_md = gr.Markdown("Search, compare, and discover the right AI tools for what you're trying to accomplish.", elem_id="header-subtitle")

            with gr.Row(elem_classes="search-shell"):
                task_input = gr.Textbox(value="", placeholder="What are you trying to do?", show_label=False, scale=5, elem_id="search-input")
                search_btn = gr.Button("Search", variant="primary", scale=1, elem_id="search-btn")

            with gr.Row(elem_classes="specialized-toggle"):
                specialized_toggle = gr.Checkbox(label="Specialized AI only", value=False)
                gr.HTML('<span class="help-tip" data-tip="Shows only AI tools built specifically for your task. Hides general-purpose tools like ChatGPT or Claude.">?</span>')

            output_html = gr.HTML()

            with gr.Row(elem_classes="compare-group"):
                compare_select = gr.CheckboxGroup(label="Select up to 5 tools to compare", choices=[], visible=False, elem_id="compare_select")
                compare_btn = gr.Button("Compare Selected", variant="primary", elem_id="compare-btn")

            with gr.Column(elem_classes="feedback-card"):
                gr.HTML('<div class="fb-info"><div class="fb-icon-wrap">💬</div><div class="fb-text-col"><div id="fb-label-html">Which one did you end up using?</div><div id="fb-sub-html">Your feedback helps us improve recommendations.</div></div></div>')
                choice = gr.Radio(show_label=False, visible=False, elem_id="fb-radio")
                with gr.Row(elem_classes="feedback-bottom"):
                    fb_output = gr.Markdown(elem_id="fb-output")
                    fb_btn = gr.Button("Submit Feedback", variant="primary", elem_id="fb-submit-btn")

        with gr.Column(scale=1, min_width=220, elem_classes="right-sidebar"):
            gr.HTML("<div class='right-card'><div class='right-title'>⚡ Popular Tasks</div>")
            for task_name in ["Create a presentation", "Write an essay", "Build a web app", "Generate AI images", "Edit a video"]:
                gr.HTML(f'<div class="right-item" style="cursor: pointer;" onclick="document.getElementById(\'search-input\').querySelector(\'textarea\').value=\'{task_name}\'; document.getElementById(\'search-input\').querySelector(\'textarea\').dispatchEvent(new Event(\'input\')); document.querySelector(\'#search-btn\').click();"><span class="popular-task-icon">⚡</span><div><div class="right-item-name">{task_name}</div></div></div>')
            gr.HTML("</div>")

            gr.HTML("<div class='right-card'><div class='right-title'>🔥 Trending This Week</div>")
            trending_tools = recommend("best ai", top_k=5)
            for i, t in enumerate(trending_tools):
                logo_url = _get_logo_url(t["url"])
                gr.HTML(f'<a href="{t["url"]}" target="_blank" class="right-item"><span class="right-item-rank">{i+1}</span><img src="{logo_url}" class="right-item-logo"><div><div class="right-item-name">{t["name"]}</div><div class="right-item-cat">{t["category"]}</div></div></a>')
            gr.HTML("</div>")

    view_outputs = [output_html, choice, compare_select, title_md, subtitle_md, home_btn, explore_btn, trending_btn, categories_btn, learn_more_btn, task_input, search_btn]

    search_btn.click(fn=search, inputs=[task_input, specialized_toggle], outputs=view_outputs)
    task_input.submit(fn=search, inputs=[task_input, specialized_toggle], outputs=view_outputs)
    specialized_toggle.change(fn=search, inputs=[task_input, specialized_toggle], outputs=view_outputs)

    compare_select.change(fn=enforce_compare_limit, inputs=compare_select, outputs=compare_select)

    compare_btn.click(fn=compare_tools, inputs=[compare_select, task_input], outputs=output_html)
    fb_btn.click(fn=feedback_handler, inputs=[task_input, choice], outputs=fb_output)

    home_btn.click(fn=show_home, inputs=[specialized_toggle], outputs=view_outputs)
    explore_btn.click(fn=show_explore, inputs=[specialized_toggle], outputs=view_outputs)
    trending_btn.click(fn=show_trending, inputs=[specialized_toggle], outputs=view_outputs)
    categories_btn.click(fn=show_categories, inputs=None, outputs=view_outputs)
    learn_more_btn.click(fn=show_about, inputs=None, outputs=view_outputs)

    app.load(fn=search, inputs=[task_input, specialized_toggle], outputs=view_outputs)

# app.queue() enables the button to show its loading spinner state while the function runs
app.queue()
app.launch()
