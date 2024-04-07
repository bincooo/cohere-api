package cohere

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/bincooo/cohere-api/common"
	"io"
	"net/http"
)

const (
	baseUrl = "https://api.cohere.ai"

	COMMAND               = "command"
	COMMAND_LIGHT         = "command-light"
	COMMAND_LIGHT_NIGHTLY = "command-light-nightly"
	COMMAND_NIGHTLY       = "command-nightly"
	COMMAND_R             = "command-r"
	COMMAND_R_PLUS        = "command-r-plus"
)

type block struct {
	Finished bool   `json:"is_finished"`
	Event    string `json:"event_type"`
	Id       string `json:"generation_id"`
	Text     string `json:"text"`
	Reason   string `json:"finish_reason"`
}

type Message struct {
	Role    string
	Message string
}

type Chat struct {
	temperature float32
	model       string
	seed        int32
	token       string
	proxies     string
}

func (c *Chat) Proxies(proxies string) {
	c.proxies = proxies
}

func New(token string, temperature float32, seed int32, model string) Chat {
	return Chat{
		token:       token,
		temperature: temperature,
		model:       model,
		seed:        seed,
	}
}

func (c *Chat) Reply(ctx context.Context, pMessages []Message, system, message string) (chan string, error) {
	payload := c.makePayload(pMessages, system, message)
	response, err := common.New().
		Proxies(c.proxies).
		Context(ctx).
		URL(fmt.Sprintf("%s/v1/chat", baseUrl)).
		Method(http.MethodPost).
		Header("Authorization", "Bearer "+c.token).
		Header("Accept-Language", "en-US,en;q=0.9").
		Header("Origin", "https://dashboard.cohere.com").
		Header("Referer", "https://dashboard.cohere.com/").
		JsonHeader().
		SetBody(payload).
		Do()
	if err != nil {
		return nil, err
	}

	if response.StatusCode != http.StatusOK {
		return nil, errors.New(response.Status)
	}

	ch := make(chan string)
	go resolve(ch, response)
	return ch, nil
}

func (c *Chat) makePayload(pMessages []Message, system string, message string) (payload map[string]interface{}) {
	if c.temperature < 0 {
		c.temperature = 0.95
	}

	payload = map[string]interface{}{
		"chat_history":      pMessages,
		"connectors":        make([]string, 0),
		"message":           message,
		"model":             c.model,
		"preamble":          system,
		"prompt_truncation": "OFF",
		"stream":            true,
		"temperature":       c.temperature,
	}

	if c.seed > 0 {
		payload["seed"] = c.seed
	}
	return payload
}

func resolve(ch chan string, response *http.Response) {
	defer close(ch)

	buf := new(bytes.Buffer)
	r := bufio.NewReader(response.Body)
	for {
		line, prefix, err := r.ReadLine()
		buf.Write(line)

		if err != nil {
			if err != io.EOF {
				ch <- fmt.Sprintf("error: %v", err)
			}
			if str := buf.String(); len(str) > 0 {
				ch <- "text: " + str
			}
			return
		}
		if prefix {
			continue
		}

		var b block
		if err = json.Unmarshal(buf.Bytes(), &b); err != nil {
			ch <- fmt.Sprintf("error: %v", err)
			return
		}
		buf.Reset()

		if b.Finished {
			return
		}

		if b.Event == "text-generation" {
			ch <- "text: " + b.Text
			continue
		}
	}
}
