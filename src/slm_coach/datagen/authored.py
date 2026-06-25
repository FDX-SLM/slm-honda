r"""Claude-authored (Opus) training seeds — the *language* is model-generated, facts are ground truth.

Spec §0: "Claude Code không bịa fact, chỉ sinh code + biến thể diễn đạt từ ground truth." Đây là chỗ
tôi (Opus) đóng góp **biến thể diễn đạt**: lời than khách viết thật (nhiều giọng, lộn xộn, typo, dài
dòng/cụt lủn) + reasoning ``<think>`` đa dạng. Mỗi seed được gán RC + ``evidence`` (cụm CÓ THẬT trong
lời than) + ``confidence``; resolution package (facts) ráp từ :mod:`slm_coach.ground_truth`, rồi cả
mẫu qua oracle (``trust_rc=True``) — nên ngôn ngữ giàu mà fact không trôi.

Mở rộng tự nhiên: thêm dict vào ``AUTHORED`` / ``AUTHORED_ABSTAIN`` và chạy lại ``gen_sft``.
"""

from __future__ import annotations

from typing import Any

from slm_coach.datagen.core import authored_abstention, authored_case
from slm_coach.datagen.records import sft_record
from slm_coach.ground_truth import RC_TO_SLICE
from slm_coach.oracle import check_resolution

# Each entry: complaint (authored), evidence (grounded fragments), think (authored), confidence.
AUTHORED: dict[str, list[dict[str, Any]]] = {
    "TCU_OFFLINE": [
        {
            "complaint": "Bought remote start last week, total waste so far. The thing sits in our condo's underground parking 24/7 and whenever I hit start it just churns and eventually says it timed out. App says my plan is active so I'm lost.",
            "evidence": ["underground parking", "churns and eventually says it timed out", "plan is active"],
            "think": "The car lives in an underground lot with no network reach and the failure is a car-side timeout, not the app hiding the feature, so the activation push most likely never reached the unit. The plan showing active rules out a never-created eligibility case. I can't see the unit's last check-in from the ticket, so I'll hold around 0.74 and flag what to verify.",
            "confidence": 0.74,
        },
        {
            "complaint": "hey so my remote climate thing won't fire up. car's been in the basement garage at work all week, like 8 days, haven't driven it once. commands spin then die. it does show active in the app fwiw",
            "evidence": ["basement garage", "haven't driven it", "commands spin then die", "active in the app"],
            "think": "Eight days parked in a basement with no signal plus commands that spin and die is a textbook connectivity timeout on the car side. Since the app still shows it active, this isn't a provisioning gap. Without the telematics check-in I'll keep confidence moderate and ask them to move it to open sky.",
            "confidence": 0.72,
        },
        {
            "complaint": "Remote lock/unlock just hangs forever. Car has been parked in an underground garage near my office, no signal down there at all. Everything looks fine on my account.",
            "evidence": ["underground garage", "no signal", "hangs forever"],
            "think": "No signal in an underground garage and a command that hangs forever points to the vehicle being offline rather than an account problem, especially since the account looks fine. I'd confirm the unit's connectivity before anything else, so confidence stays around 0.7.",
            "confidence": 0.7,
        },
        {
            "complaint": "I am extremely frustrated. I paid for Vehicle Finder and it times out every single time I try it. The car has not moved from the underground lot in over a week. It clearly shows as active though.",
            "evidence": ["times out every single time", "has not moved from the underground lot", "shows as active"],
            "think": "A feature that shows active but times out every time, on a car that hasn't left an underground lot in a week, is connectivity-bound, not an entitlement issue. The car simply isn't reachable to receive the push. I'll set ~0.76 given the strong, consistent cues and list the connectivity check.",
            "confidence": 0.76,
        },
        {
            "complaint": "remote start = useless rn. garage is like 3 levels down, zero bars. press the button, spinner, timeout. subscription active per the app though so wtf",
            "evidence": ["garage is like 3 levels down", "zero bars", "spinner, timeout", "subscription active"],
            "think": "Three levels underground with zero bars and a spinner-then-timeout is the car failing to receive the command, not the app failing to show the feature, and the subscription is active. That's an offline telematics unit. Confidence around 0.75; the fix is restoring signal.",
            "confidence": 0.75,
        },
        {
            "complaint": "My connected services won't respond. The vehicle's been sitting in a covered parking structure for days and there's basically no reception in there. The app itself shows everything turned on.",
            "evidence": ["covered parking structure for days", "no reception", "won't respond", "shows everything turned on"],
            "think": "The car is in a covered structure with no reception and the app shows everything on, so the feature is provisioned but the vehicle can't be reached. The non-response is on the car side. I'd verify the unit's last contact and keep confidence near 0.7.",
            "confidence": 0.7,
        },
        {
            "complaint": "ok weird one. remote engine start spins and times out but ONLY since I started parking in the new underground spot. used to work in my driveway. shows active.",
            "evidence": ["spins and times out", "underground spot", "shows active"],
            "think": "It worked in an open driveway and only broke after moving to an underground spot, which strongly implicates lost cellular reception rather than entitlement or cache. The timeout is car-side and the feature shows active. That environmental correlation pushes me to about 0.78.",
            "confidence": 0.78,
        },
        {
            "complaint": "Subscription is paid and active, I checked. But the car won't take any remote commands — they just time out. It's been garaged underground while I'm traveling.",
            "evidence": ["active", "won't take any remote commands", "time out", "garaged underground"],
            "think": "Paid and active subscription with commands timing out while the car is garaged underground is connectivity, not provisioning. The vehicle is unreachable. I'll keep ~0.73 and ask them to bring it into the open.",
            "confidence": 0.73,
        },
        {
            "complaint": "the app keeps telling me the remote command failed. timeout. my truck hasnt been driven in like 10 days, its in a parking garage downtown. signal there is garbage. feature shows on.",
            "evidence": ["remote command failed", "timeout", "hasnt been driven in like 10 days", "parking garage", "feature shows on"],
            "think": "Ten days idle in a downtown parking garage with bad signal and repeated command timeouts is a vehicle that can't receive pushes; the feature itself shows on. This is offline telematics rather than a stale app view. Confidence ~0.74.",
            "confidence": 0.74,
        },
        {
            "complaint": "Remote start times out. Car's in long-term airport parking, underground level, been there since my trip started. App says active.",
            "evidence": ["times out", "underground level", "App says active"],
            "think": "Long-term underground airport parking is a classic dead-zone, and a timeout with the app showing active means the car isn't getting the command. Not an eligibility or cache problem. I'll hold ~0.74 pending the connectivity check.",
            "confidence": 0.74,
        },
        {
            "complaint": "so the climate preconditioning won't kick on. tried a bunch of times, always times out. the car is parked deep in a concrete garage with no bars. it does say active though, that's the confusing part",
            "evidence": ["won't kick on", "always times out", "concrete garage with no bars", "say active"],
            "think": "A concrete garage with no bars plus repeated timeouts and an active status means the vehicle is provisioned but offline, so the push isn't landing. The confusion the customer feels is exactly the active-but-unreachable pattern. Confidence about 0.73.",
            "confidence": 0.73,
        },
        {
            "complaint": "Paid customer here. Remote features unresponsive. The car has been parked underground all week and there is no cellular down there. Everything reads active on my end.",
            "evidence": ["unresponsive", "parked underground all week", "no cellular", "reads active"],
            "think": "No cellular underground all week with unresponsive remote features that still read active is a vehicle offline, not a billing or eligibility failure. The entitlement reached the platform but not the car. I'll keep ~0.75.",
            "confidence": 0.75,
        },
        {
            "complaint": "remote unlock won't go through, just keeps spinning til it errors out with a timeout. left the car in my building's lower garage for two weeks. shows active in app.",
            "evidence": ["keeps spinning", "errors out with a timeout", "lower garage for two weeks", "shows active"],
            "think": "Two weeks in a lower garage and a spinner that errors out on timeout, with the app showing active, is the vehicle being unreachable rather than the entitlement missing. I'd confirm the unit's connectivity and keep confidence near 0.74.",
            "confidence": 0.74,
        },
        {
            "complaint": "Honestly thought I broke something. Remote start subscription is active but the car just sits there and the app spins and times out. It's been in a basement parking garage and hasn't moved.",
            "evidence": ["subscription is active", "spins and times out", "basement parking garage and hasn't moved"],
            "think": "An active subscription with a spin-and-timeout on a car that hasn't moved from a basement garage points to lost connectivity, not a customer error and not a provisioning gap. The vehicle can't receive the push. Confidence ~0.74.",
            "confidence": 0.74,
        },
    ],
    "ENTITLEMENT_CACHE_STALE": [
        {
            "complaint": "super annoying. I can literally see my plan is active on the website, like it's right there, but the phone app insists I don't have it. paid two days ago.",
            "evidence": ["active on the website", "app insists I don't have it"],
            "think": "The entitlement plainly exists because the website shows it active, yet the app renders it as missing, which is the classic mismatched-view symptom of a stale cache rather than a connectivity or eligibility failure. I'd confirm the record and force a refresh; confidence around 0.72.",
            "confidence": 0.72,
        },
        {
            "complaint": "had it working tuesday, gone wednesday, then it came back thursday after I logged out and back in. it's flickering on and off basically. account looks normal.",
            "evidence": ["working tuesday, gone wednesday", "logged out and back in", "flickering on and off"],
            "think": "A feature that worked, vanished, and returned after a re-login, flickering on and off, is a cache consistency problem, not a vehicle-side timeout or a never-created entitlement. Re-login clearing it is a strong tell. Confidence ~0.74.",
            "confidence": 0.74,
        },
        {
            "complaint": "The feature shows up green and active on the web portal but the HondaLink app says I need to buy it. That's contradictory right? I already paid.",
            "evidence": ["active on the web portal", "app says I need to buy it"],
            "think": "Active on the web portal but the app prompting to buy is contradictory only if you assume both read the same fresh source; it's the cache serving the app a stale view while the entitlement exists. This isn't eligibility because it does show active somewhere. I'll hold ~0.7 and confirm the record then invalidate the cache.",
            "confidence": 0.7,
        },
        {
            "complaint": "my remote climate just disappeared from the app out of nowhere. comes back sometimes if I refresh a few times. everything looks fine on my account page.",
            "evidence": ["disappeared from the app", "comes back sometimes if I refresh", "looks fine on my account"],
            "think": "Intermittent disappearance that a refresh brings back, with the account page looking fine, is a display/sync issue from a stale cache. A connectivity problem would be car-side, and eligibility would never show it anywhere. Confidence ~0.73.",
            "confidence": 0.73,
        },
        {
            "complaint": "works on the website, doesn't work in the app. classic. it was fine yesterday and then suddenly stopped this morning. logging out helps for a minute then it's back to missing.",
            "evidence": ["works on the website, doesn't work in the app", "fine yesterday and then suddenly stopped", "logging out helps"],
            "think": "Works on web, not in app, after a sudden change overnight, with logging out giving a brief fix, is a stale cached entitlement in the app layer. The worked-then-stopped plus relogin pattern rules out eligibility and TCU offline. Confidence ~0.75.",
            "confidence": 0.75,
        },
        {
            "complaint": "Account dashboard clearly lists my subscription as active. App on my Pixel does not show it at all. Re-logging in sometimes makes it pop up briefly.",
            "evidence": ["active", "App", "does not show it", "Re-logging in sometimes makes it pop up"],
            "think": "The dashboard lists it active while the app doesn't, and a re-login briefly surfaces it, which is a cache that hasn't picked up the entitlement rather than a vehicle or eligibility issue. I'd confirm the record then invalidate the cache; confidence ~0.73.",
            "confidence": 0.73,
        },
        {
            "complaint": "it's so inconsistent. one minute the feature is there, next minute it's gone. paid for it last week. when I check online it's definitely active.",
            "evidence": ["so inconsistent", "there, next minute it's gone", "online it's definitely active"],
            "think": "Inconsistent appear/disappear behavior with the online view definitely active is a cache sync problem; the entitlement exists but the app intermittently serves a stale state. Not eligibility, since it does show active. Confidence ~0.72.",
            "confidence": 0.72,
        },
        {
            "complaint": "Why does the website say I have remote start but the app keeps showing it as off? It worked before. I even reinstalled the app, no change.",
            "evidence": ["website say I have remote start", "app keeps showing it as off", "worked before"],
            "think": "Web shows the entitlement, the app shows it off, and it worked before, which is a stale cached view rather than a missing entitlement or an offline car. The reinstall not helping fits a server-side cache, not the client. Confidence ~0.7.",
            "confidence": 0.7,
        },
        {
            "complaint": "remote features keep blinking out in the app. active on my online account 100%. started yesterday, was totally fine before that.",
            "evidence": ["blinking out in the app", "active on my online account", "fine before"],
            "think": "Features blinking out in the app while the online account stays active, starting suddenly after being fine, is a cache that fell out of sync. A car-side problem would be a timeout, not a flicker in the app view. Confidence ~0.73.",
            "confidence": 0.73,
        },
        {
            "complaint": "I think there's a sync issue? The app says subscribe but my account online shows it active and I was using it fine two days ago. Refreshing helps temporarily.",
            "evidence": ["app says subscribe", "account online shows it active", "using it fine two days ago", "Refreshing helps temporarily"],
            "think": "The customer themselves suspects sync, and the signals agree: active online, prompting in the app, worked recently, refresh helps briefly. That is stale cache, not eligibility, because the entitlement is clearly present. I'll keep ~0.74 and invalidate the cache after confirming the record.",
            "confidence": 0.74,
        },
        {
            "complaint": "feature was working all month then poof, gone from the app today. web still shows active. logged out, logged in, showed up for a sec, gone again.",
            "evidence": ["working all month then poof, gone from the app", "web still shows active", "logged out, logged in, showed up for a sec"],
            "think": "A month of working then a sudden disappearance in the app while web stays active, with a re-login flashing it briefly, is a cache that lost the entitlement view. Strong worked-then-stopped and relogin cues. Confidence ~0.75.",
            "confidence": 0.75,
        },
        {
            "complaint": "Remote start shows active on the Honda website but not in the app, very frustrating, it comes and goes. Paid customer, been like this on and off since yesterday.",
            "evidence": ["active on the Honda website but not in the app", "comes and goes", "since yesterday"],
            "think": "Active on web, not in app, coming and going since yesterday is intermittent stale-cache behavior; the entitlement exists so it isn't eligibility, and it's an app-view issue rather than a car timeout. Confidence ~0.72.",
            "confidence": 0.72,
        },
    ],
    "ELIGIBILITY_RULE_CONFLICT": [
        {
            "complaint": "I paid for the Touring plan on my CR-V (2025, I'm in Seattle so US-West) and the app STILL prompts me to subscribe every single time. I already paid you guys. This is persistent, not a glitch.",
            "evidence": ["paid for the Touring plan", "CR-V", "US-West", "prompts me to subscribe every single time", "persistent"],
            "think": "Payment went through but the app persistently prompts to subscribe for a CR-V Touring in US-West, which is a region/trim/plan combination that an eligibility rule can reject, leaving no entitlement created. It's persistent rather than flickering, which argues against stale cache. I can't see the eligibility decision from the ticket, so confidence ~0.72.",
            "confidence": 0.72,
        },
        {
            "complaint": "money came out of my account, confirmed on my statement, but the feature never activated. app just keeps asking me to buy it again. it's a premium plan on a kinda niche trim.",
            "evidence": ["money came out of my account", "never activated", "keeps asking me to buy it again", "premium plan", "niche trim"],
            "think": "Confirmed payment with no activation and the app repeatedly asking to buy again, on a premium plan with a niche trim, fits an eligibility rule blocking a region/trim/plan combo so the entitlement was never created. Nothing shows active anywhere, which separates it from a stale cache. Confidence ~0.7.",
            "confidence": 0.7,
        },
        {
            "complaint": "So I have a Civic, up in Canada, and I went for the Touring tier. Charged me fine. The app acts like I never subscribed at all, been days now, every time the same.",
            "evidence": ["Civic", "Canada", "Touring", "acts like I never subscribed", "been days now"],
            "think": "A Canada Civic Touring that charged fine but where the app acts like there's no subscription for days is a likely eligibility rejection of that combo, so no entitlement was ever created. The persistence over days and nothing showing active rules out cache. Confidence ~0.71.",
            "confidence": 0.71,
        },
        {
            "complaint": "why does it keep telling me to subscribe when I ALREADY PAID. ugh. it's a top-tier plan, limited region thing. immediate and constant, never works for a second.",
            "evidence": ["keep telling me to subscribe", "ALREADY PAID", "top-tier plan, limited region", "immediate and constant"],
            "think": "Immediate, constant subscribe prompts after payment on a top-tier, limited-region plan are consistent with an eligibility rule rejecting the combination, so the entitlement never got created. It never works even briefly, which is the opposite of a flickering cache. Confidence ~0.72.",
            "confidence": 0.72,
        },
        {
            "complaint": "Paid for Elite on my CR-V out in Oregon — wait, no, I think I actually picked Touring. Anyway it won't activate and keeps prompting purchase. Been like that since I bought it.",
            "evidence": ["CR-V", "Oregon", "Touring", "won't activate and keeps prompting purchase", "since I bought it"],
            "think": "A CR-V Touring in Oregon (US-West) that won't activate and keeps prompting purchase since the moment of buying points to an eligibility rule blocking that combo, so no entitlement was created. Persistent from purchase, nothing active, which isn't cache. Confidence ~0.7.",
            "confidence": 0.7,
        },
        {
            "complaint": "I purchased the subscription, payment cleared, but the app won't let me use it and shows the Subscribe button like I never bought anything. It's a region-restricted premium combo.",
            "evidence": ["payment cleared", "shows the Subscribe button", "never bought anything", "region-restricted premium combo"],
            "think": "Cleared payment but a persistent Subscribe button on a region-restricted premium combo is an eligibility block that prevented entitlement creation, so nothing propagates to the app. It shows active nowhere, separating it from a stale cache. Confidence ~0.71.",
            "confidence": 0.71,
        },
        {
            "complaint": "third time calling about this. paid for touring, it asks me to subscribe again, every time, no exceptions. my trim/region combo is a bit unusual I guess. never see it active anywhere.",
            "evidence": ["paid for touring", "asks me to subscribe again, every time", "trim/region combo is a bit unusual", "never see it active anywhere"],
            "think": "Repeat calls, a Touring payment, constant subscribe prompts, an unusual trim/region combo, and never seeing it active anywhere all line up with an eligibility rule rejecting the combination. The 'active nowhere' detail is the key separator from cache. Confidence ~0.73.",
            "confidence": 0.73,
        },
        {
            "complaint": "App keeps asking me to buy the plan I already bought. Receipt and everything. It's a newer trim in a limited market. Happens instantly, every single time, dead consistent.",
            "evidence": ["asking me to buy the plan I already bought", "newer trim in a limited market", "instantly, every single time, dead consistent"],
            "think": "Being asked to buy an already-bought plan, instantly and consistently, on a newer trim in a limited market is an eligibility rejection of that combo with no entitlement created. The dead consistency rules out intermittent cache behavior. Confidence ~0.72.",
            "confidence": 0.72,
        },
        {
            "complaint": "i paid, i have the email receipt, but the app still wants me to subscribe. it's a touring package on a CR-V in the western region. it's been like this nonstop.",
            "evidence": ["i paid", "still wants me to subscribe", "touring package on a CR-V in the western region", "nonstop"],
            "think": "A paid Touring package on a CR-V in the western region with a nonstop subscribe prompt is the trap combination an eligibility rule rejects, so no entitlement was created. Nonstop and never active points away from cache. Confidence ~0.72.",
            "confidence": 0.72,
        },
        {
            "complaint": "Charged correctly but no access. The Subscribe prompt won't go away no matter what. My plan/region/vehicle mix is apparently on the unusual side. This has been constant since day one.",
            "evidence": ["Charged correctly but no access", "Subscribe prompt won't go away", "plan/region/vehicle mix", "unusual", "constant since day one"],
            "think": "Charged but no access with a Subscribe prompt that won't go away on an unusual plan/region/vehicle mix, constant since day one, is an eligibility conflict preventing entitlement creation. Constant-from-day-one is unlike a cache that would have worked at some point. Confidence ~0.71.",
            "confidence": 0.71,
        },
        {
            "complaint": "ok I'm mad. paid for the premium connected plan, app still says subscribe, every time I open it. I drive a less common trim and I'm in a smaller region. zero days where it worked.",
            "evidence": ["paid for the premium connected plan", "app still says subscribe", "less common trim", "smaller region", "zero days where it worked"],
            "think": "A premium plan that was paid for but always shows subscribe, on a less common trim in a smaller region with zero days of working, indicates an eligibility rule rejecting the combo so the entitlement was never made. The complete absence of any working period separates it from cache. Confidence ~0.73.",
            "confidence": 0.73,
        },
        {
            "complaint": "Payment confirmed by my bank. Feature not granted. App repeatedly prompts to subscribe. It's a top trim with a region-limited rollout. Persistent, immediate, never intermittent.",
            "evidence": ["Payment confirmed", "not granted", "prompts to subscribe", "top trim with a region-limited rollout", "Persistent, immediate, never intermittent"],
            "think": "Bank-confirmed payment, no grant, repeated subscribe prompts, a top trim under a region-limited rollout, and explicitly never intermittent all point to an eligibility rule blocking the combo so no entitlement formed. The 'never intermittent' is the clean separator from cache. Confidence ~0.73.",
            "confidence": 0.73,
        },
    ],
}

# Ambiguous / out-of-catalog → abstain. Authored to be genuinely under-specified or off-catalog.
AUTHORED_ABSTAIN: list[dict[str, Any]] = [
    {
        "complaint": "nothing works. i paid. please just fix it. i don't even know what's wrong, it just won't do the thing.",
        "evidence": [],
        "think": "There's no distinguishing detail here: I don't know whether the feature shows active anywhere, whether the car is offline, or whether the app keeps prompting to subscribe. With no cue to separate the causes I won't commit; I keep confidence low and route to a human with a short checklist.",
    },
    {
        "complaint": "I keep getting error 403 whenever I tap the remote features. logged in fine otherwise. what does that even mean",
        "evidence": ["error 403"],
        "think": "A 403 on the remote features is an access/permissions error outside the three onboarding root causes I diagnose, so I shouldn't force a cache/TCU/eligibility label. I'll route this to a human and note the 403 for them to investigate.",
    },
    {
        "complaint": "my subscription is broken somehow, can you look into it? my account number's on file.",
        "evidence": [],
        "think": "This is too vague to diagnose: 'broken somehow' gives me no cue about where the feature shows active, whether the car responds, or what the app displays. I'll abstain and ask a human to gather specifics rather than guess.",
    },
    {
        "complaint": "the app crashes the second I open the remote tab. can't even get to the feature to try it.",
        "evidence": ["app crashes"],
        "think": "An app crash on open is a client stability problem, not one of the entitlement root causes I resolve, and I can't observe any cue about the entitlement itself. I'll route to a human rather than force a root cause.",
    },
    {
        "complaint": "honestly I just want a refund, this whole thing's a mess and I might've been charged twice.",
        "evidence": ["refund", "charged twice"],
        "think": "This is a billing/refund dispute, which is outside the onboarding diagnosis catalog, and there's no cue about how the feature itself behaves. I'll abstain and hand it to a human for the billing path.",
    },
    {
        "complaint": "it doesn't work right. sometimes. idk. can someone call me",
        "evidence": [],
        "think": "'Doesn't work right, sometimes' with no further detail gives me nothing to separate a cache, connectivity, or eligibility cause. Forcing a label would be a guess, so I'll keep confidence low and route to a human.",
    },
    {
        "complaint": "the app logs me out constantly, like every few minutes. super annoying.",
        "evidence": ["logs me out constantly"],
        "think": "Constant logouts are an authentication/session issue rather than one of the three entitlement root causes, and there's no cue about the feature's state. I'll route this to a human instead of forcing a diagnosis.",
    },
    {
        "complaint": "there's some update stuck on the car screen and now I'm not sure if my subscription stuff is affected. confusing.",
        "evidence": ["update stuck"],
        "think": "A stuck on-vehicle software update is outside my catalog and the customer isn't actually reporting a specific entitlement symptom, just uncertainty. Without a distinguishing cue I'll abstain and route to a human.",
    },
    {
        "complaint": "Hi, my connected services aren't behaving. Not sure how to describe it. Can you check my account?",
        "evidence": [],
        "think": "'Aren't behaving' with an explicit 'not sure how to describe it' gives me no cue to pick between the root causes. The honest move is to abstain and have a human pull the specifics.",
    },
    {
        "complaint": "I got a message saying access denied or something? for the remote stuff. weird.",
        "evidence": ["access denied"],
        "think": "An access-denied message is a permissions/token error outside the three onboarding root causes, and there's no cue about whether the entitlement shows active anywhere. I'll route to a human rather than label it.",
    },
    {
        "complaint": "Something is off with my plan. It's not doing what it should. Please advise.",
        "evidence": [],
        "think": "This is under-specified: no indication of where the feature shows active, whether the car responds, or whether the app prompts to subscribe. I won't commit to a cause and will route to a human with a checklist.",
    },
    {
        "complaint": "everything was great then it just... isn't. no idea. help",
        "evidence": [],
        "think": "'Just isn't' with no detail is not enough to distinguish a stale cache from connectivity or eligibility. I'll keep confidence low, list what to confirm, and route to a human.",
    },
    # --- H3: not actually a fault → don't invent a root cause ---
    {
        "complaint": "Just confirming — my remote start works great. Does my subscription renew next month?",
        "evidence": [],
        "think": "This isn't a fault report at all: the customer says the feature works great and is asking a billing/renewal question. There is no symptom to diagnose, so I must not fabricate a root cause; I route this to a human (billing) rather than force one of the failure classes.",
    },
    {
        "complaint": "Everything's working fine, I just wanted to ask how I upgrade to the Touring plan.",
        "evidence": [],
        "think": "Nothing is broken here — it's a sales/upgrade question, not an entitlement failure. With no fault symptom there is no root cause to assign, so I route to a human instead of inventing one.",
    },
    {
        "complaint": "No issues, just checking — will my remote features keep working if I switch phones?",
        "evidence": [],
        "think": "The customer states there are no issues and is asking a general how-to question. There is no malfunction to diagnose, so I abstain from assigning a root cause and route the question to a human.",
    },
]


# --- E: thin-cue cases → calibrated LOWER confidence (~0.6) -----------------------------------
AUTHORED_THIN: dict[str, list[dict[str, Any]]] = {
    "ELIGIBILITY_RULE_CONFLICT": [
        {
            "complaint": "The app keeps asking me to subscribe again.",
            "evidence": ["keeps asking me to subscribe again"],
            "think": "The only cue is the app repeatedly asking to subscribe, which leans toward an eligibility block rather than a cache or connectivity issue. But there's no detail on the vehicle, region, or whether it ever showed active, so I stay tentative at about 0.6 and lean on what to confirm.",
            "confidence": 0.6,
        },
        {
            "complaint": "It won't let me use the plan I paid for, still says subscribe.",
            "evidence": ["still says subscribe", "plan I paid for"],
            "think": "Paid but still prompted to subscribe points toward eligibility, yet with no combo or persistence detail I can't be confident; I'll keep it around 0.6 and list the eligibility checks.",
            "confidence": 0.6,
        },
    ],
    "ENTITLEMENT_CACHE_STALE": [
        {
            "complaint": "It shows active on the site but not in the app.",
            "evidence": ["active on the site but not in the app"],
            "think": "Active on the site but not the app suggests a stale cached view, but it's a single data point with no history of it working or flickering, so I keep confidence modest near 0.62 and confirm the record before invalidating.",
            "confidence": 0.62,
        },
    ],
    "TCU_OFFLINE": [
        {
            "complaint": "Remote start times out and the car's in a garage.",
            "evidence": ["times out", "car's in a garage"],
            "think": "A timeout with the car in a garage hints at lost connectivity, but I have no detail on how long it's been parked or whether it shows active, so I stay around 0.62 and verify the unit's reachability.",
            "confidence": 0.62,
        },
    ],
}

# --- H1: conflicting cues across RCs → differential, pick the dominant branch ------------------
_CONFLICT_DIFF = [
    {"rc": "TCU_OFFLINE", "likelihood": "high", "why": "a car-side timeout while parked with no signal"},
    {"rc": "ENTITLEMENT_CACHE_STALE", "likelihood": "medium", "why": "the 'active but won't turn on' wording could be a stale view, but the garage and car-side stall point to the vehicle being offline"},
]
AUTHORED_CONFLICT: list[dict[str, Any]] = [
    {
        "leading": "TCU_OFFLINE",
        "complaint": "The app says it's active but it won't turn on — and my car's also been in the garage all week.",
        "evidence": ["says it's active but it won't turn on", "in the garage all week"],
        "think": "There are two readings here: 'active but won't turn on' could be a stale cache, but the car being in the garage all week strongly suggests the vehicle is offline and the command can't reach it. Both deserve a place in the differential; I weight TCU offline higher because the failure is a car-side stall, and keep confidence moderate (~0.62) pending the connectivity check.",
        "confidence": 0.62,
        "differential": _CONFLICT_DIFF,
    },
    {
        "leading": "TCU_OFFLINE",
        "complaint": "Shows active, just won't respond. The car's been sitting in an underground lot for two weeks now.",
        "evidence": ["Shows active, just won't respond", "underground lot for two weeks"],
        "think": "An active-but-unresponsive feature could read as cache, but two weeks in an underground lot makes lost connectivity the stronger explanation for a non-response on the car side. I list both, lead with TCU offline, and hold around 0.64.",
        "confidence": 0.64,
        "differential": [
            {"rc": "TCU_OFFLINE", "likelihood": "high", "why": "two weeks underground with no response is a car-side connectivity stall"},
            {"rc": "ENTITLEMENT_CACHE_STALE", "likelihood": "medium", "why": "active-but-not-working could be a stale view, but that wouldn't explain the long underground parking"},
        ],
    },
    {
        "leading": "TCU_OFFLINE",
        "complaint": "I can see it's active but it won't work, the car has no signal in the basement parking where it's parked.",
        "evidence": ["it's active but it won't work", "no signal in the basement parking"],
        "think": "Active-but-not-working hints at cache, yet 'no signal in the basement parking' is a direct connectivity cue that better explains why the car won't act. I keep both in the differential, lead with TCU offline, confidence ~0.63.",
        "confidence": 0.63,
        "differential": _CONFLICT_DIFF,
    },
]

# --- H2: angry, escalated paid customer → still RC-5, churn high, correct policy ---------------
AUTHORED_ESCALATED: list[dict[str, Any]] = [
    {
        "leading": "ELIGIBILITY_RULE_CONFLICT",
        "complaint": "I've called three times now. I want a full refund and a free year or I'm leaving. My CR-V Touring in California still won't activate even though I paid.",
        "evidence": ["called three times", "CR-V Touring in California", "still won't activate", "paid"],
        "think": "Despite the heat, the diagnosis is unchanged: a paid CR-V Touring in California that won't activate is the eligibility combo being rejected, so no entitlement was created. The repeated calls and threat to leave make churn high, so I'll follow policy and proactively offer the one-month credit — without caving to the full-refund-and-free-year demand, which isn't policy. Confidence ~0.72.",
        "confidence": 0.72,
        "churn_override": {"level": "high", "why": "paid customer, called 3+ times, threatening to leave; feels cheated"},
    },
]


def generate_authored() -> list[dict[str, Any]]:
    """Materialize the Claude-authored seeds into oracle-passing SFT records.

    Returns:
        A list of SFT records. Each authored case is oracle-checked with ``trust_rc=True`` (the
        author owns the cue→RC mapping); the honesty rules still run. Raises if a seed fails, so
        authoring mistakes surface immediately.
    """
    records: list[dict[str, Any]] = []
    from slm_coach.datagen.core import assistant_content
    from slm_coach.ground_truth import SYSTEM_PROMPT

    def _emit(case: Any, slice_tag: str, label: str) -> None:
        result = check_resolution(case.complaint, case.think, case.resolution, trust_rc=True)
        if not result.ok:
            raise ValueError(f"authored {label} seed failed oracle: {result.failures}\n{case.complaint}")
        records.append(
            sft_record(
                f"sft-auth-{len(records):05d}",
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case.complaint},
                    {"role": "assistant", "content": assistant_content(case.think, case.resolution)},
                ],
                slice_tag=slice_tag,
            )
        )

    # Core RC cases + thin-cue (E) + escalated (H2), all per concrete RC.
    for rc in AUTHORED:
        seeds = [*AUTHORED[rc], *AUTHORED_THIN.get(rc, [])]
        seeds += [s for s in AUTHORED_ESCALATED if s["leading"] == rc]
        for seed in seeds:
            case = authored_case(
                rc, seed["complaint"], seed["evidence"], seed["think"],
                confidence=seed["confidence"],
                differential=seed.get("differential"),
                churn_override=seed.get("churn_override"),
            )
            _emit(case, RC_TO_SLICE[rc], rc)

    # H1: conflicting-cue differential cases (slice = differential).
    for seed in AUTHORED_CONFLICT:
        case = authored_case(
            seed["leading"], seed["complaint"], seed["evidence"], seed["think"],
            confidence=seed["confidence"], differential=seed["differential"],
        )
        _emit(case, "differential", "conflict")

    for seed in AUTHORED_ABSTAIN:
        case = authored_abstention(seed["complaint"], seed["evidence"], seed["think"])
        _emit(case, "abstention", "abstain")
    return records


def authored_counts() -> dict[str, int]:
    """Per-slice authored-seed counts (for reporting)."""
    counts = {rc: len(seeds) + len(AUTHORED_THIN.get(rc, [])) for rc, seeds in AUTHORED.items()}
    for s in AUTHORED_ESCALATED:
        counts[s["leading"]] = counts.get(s["leading"], 0) + 1
    counts["conflict(differential)"] = len(AUTHORED_CONFLICT)
    counts["abstention"] = len(AUTHORED_ABSTAIN)
    return counts
