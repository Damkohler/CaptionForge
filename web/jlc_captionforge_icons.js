import { app } from "/scripts/app.js";

const ICON_SIZE = 12;

const CAPTIONFORGE_NODE_NAMES = new Set([
    "JLC_QwenCaption",
    "JLC_JoyCaption",
    "JLC_QwenCaptionLite",
    "JLC_JoyCaptionLite",
    "JLC_CaptionForgeClaimExtractor",
]);

const iconImage = new Image();
iconImage.src = new URL(
    "./assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png",
    import.meta.url
).href;

function isCaptionForgeNode(nodeData) {
    return Boolean(
        nodeData &&
        typeof nodeData.name === "string" &&
        CAPTIONFORGE_NODE_NAMES.has(nodeData.name)
    );
}

app.registerExtension({
    name: "JLC.CaptionForge.Icons",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isCaptionForgeNode(nodeData)) {
            return;
        }

        if (nodeType.prototype.__jlcCaptionForgeIconApplied) {
            return;
        }

        nodeType.prototype.__jlcCaptionForgeIconApplied = true;

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;

        nodeType.prototype.onDrawForeground = function (ctx) {
            if (originalOnDrawForeground) {
                originalOnDrawForeground.apply(this, arguments);
            }

            try {
                if (!iconImage.complete || iconImage.naturalWidth <= 0) {
                    return;
                }

                ctx.save();
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";

                const x = ICON_SIZE + 18;
                const y = -(ICON_SIZE + 9);

                ctx.drawImage(iconImage, x, y, ICON_SIZE, ICON_SIZE);
                ctx.restore();
            } catch (err) {
                try {
                    ctx.restore();
                } catch (_) {
                    // no-op
                }
                console.warn("[CaptionForge Icons] draw skipped:", err);
            }
        };
    },
});